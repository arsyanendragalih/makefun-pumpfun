#!/usr/bin/env python3
"""pumpfun-top5-forecast-01 — top-5 pump.fun coin forecaster.

Reads the house snapshots in /input (history.ndjson + task.json) and writes one
deterministic forecast to /output/forecast.json: for each of the 5 places, the
probability that each named coin is at that place at the answer reads, the top-5
guess, and each place's market-cap size-bin probabilities.

Model
-----
Each known coin shares a market cap (log10 of cap in SOL) that carries momentum
but mean-reverts and can collapse, the two forces every pump.fun coin faces in
its first day. For each coin we estimate, from its 10-minute marks:
  L      current log10(cap in SOL)
  m      recent velocity (dex/hour), how fast it has just been moving
and, from its venue and age, its durability. We then run a Monte Carlo over the
horizon: per scenario each coin either survives (drawn by its durability) into
one of three regimes -- climb (momentum carries, capped), hold (steady, slight
overall fade), or fade (fast decline out of contention) -- or collapses off
the chart. Each coin's log10-cap at an answer read is its regime's drift plus a
normal diffusion. Repeating over scenarios gives, for every place, how often
each coin holds it, and the distribution of that place's market cap.

Newcomers and far-fades (coins the history never shows, launched after the cut)
are not nameable, so every place keeps an unnamed share calibrated from the
uncertainty -- shared by the scorer among the chart coins not named there.

Deterministic: one fixed RNG seed; two runs on the same history write the same
bytes. stdlib only, no network.
"""

import json
import math
import os
import random
from collections import OrderedDict

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 2026111202          # fixed -> byte-identical re-runs
N_SCEN = 2400              # scenarios
MICRO = 1_000_000
MIN_P = 0.0001             # scorer floor
EMIT_FLOOR = 0.004         # probability rows below this are left unnamed

# ---- regime / durability parameters (domain priors for a 7-13 h pump.fun day)
SIGMA_BASE = 0.34          # dex / sqrt(hour) for a pooled / established coin
SIGMA_YOUNG = 0.47         # ... for a young / curve coin
HOLD_DRIFT = -0.045        # dex/hour: steady market overall drifts down a little
FADE_DRIFT = -0.55         # dex/hour: a fading coin leaves the top quickly
CLIMB_MOM = 0.16           # how much momentum carries for a climber (dex/h per dex/h)
CLIMB_FLOOR = 0.05         # climbers always grow at least this much (dex/h)
CONTEND = 2.4              # a coin below 10^CONTEND SOL is out of the top fight
SURV_POOL = 0.95           # durability bonus for a graduated (pool) coin
TARGET_NAMED = {1: 0.58, 2: 0.56, 3: 0.54, 4: 0.52, 5: 0.50}
NEWCOMER_TAIL = 0.05       # share of a place's size mass put on a monster newcomer
SIZE_UNNAMED = 0.02        # sizes kept to this under 1


def size_of(lamports):
    """log10 of a cap in SOL, as the scorer takes it."""
    return math.log10(float(int(lamports))) - 9.0


def sig(x):
    if x >= 0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    e = math.exp(x)
    return e / (1.0 + e)


def survival_p(L, pool, age_h):
    """Probability the coin is still in contention at the answer reads."""
    s = (L - 3.1) + (SURV_POOL if pool else 0.0) + 0.30 * min(1.0, age_h / 12.0)
    return min(0.97, max(0.06, sig(s)))


def regime_probs(m):
    """Survivor odds of climb / hold / fade from recent velocity m (dex/h)."""
    climb_w = 1.25 * sig(3.0 * m)
    fade_w = 1.25 * sig(3.0 * (-m)) + 0.15 * sig(4.0 * (-m)) ** 2
    hold_w = 0.85
    t = climb_w + fade_w + hold_w
    return climb_w / t, hold_w / t, fade_w / t


def main():
    task = json.load(open(os.path.join(INPUT_DIR, "task.json"), "r"))
    places = int(task["places"])
    edges = [float(e) for e in task["size"]["edges"]]
    nbins = len(edges) + 1
    reads = task["reads"]

    marks = []
    for line in open(os.path.join(INPUT_DIR, "history.ndjson"), "r"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("ok") and isinstance(d.get("rows"), list):
            marks.append(d)
    marks.sort(key=lambda d: d["at"])
    if not marks:
        raise RuntimeError("history holds no ok mark")

    last = marks[-1]
    T_last = last["at"]

    def hours_between(a, b):
        from datetime import datetime
        def p(t):
            return datetime.fromisoformat(t.replace("Z", "+00:00"))
        return (p(b) - p(a)).total_seconds() / 3600.0

    # ---- per-coin log-cap time series, in file order
    series = OrderedDict()
    for mi, m in enumerate(marks):
        t = m["at"]
        for r in m["rows"]:
            mint = r["mint"]
            rec = series.setdefault(mint, {"t": [], "L": []})
            rec["t"].append(t)
            rec["L"].append(size_of(r["market_cap_lamports"]))

    # ---- reduce to features at the last mark
    rng = random.Random(SEED)

    last_rows = sorted(last["rows"], key=lambda r: r["rank"])
    coins = []  # dict per coin: mint, L, m, pool, survive, sigma, created
    for r in last_rows:
        mint = r["mint"]
        L = size_of(r["market_cap_lamports"])
        pool = r.get("venue") == "pool"
        age_h = hours_between(r.get("created_at", T_last), T_last)
        # velocity from the recent series
        rec = series.get(mint, {"t": [T_last], "L": [L]})
        ts, Ls = rec["t"], rec["L"]
        # use up to the last ~6 marks
        if len(ts) > 1:
            ts = ts[-6:]
            Ls = Ls[-6:]
            if len(ts) >= 3:
                # least squares slope (dex / hour)
                xs = [hours_between(ts[0], t) for t in ts]
                xm = sum(xs) / len(xs)
                lm = sum(Ls) / len(Ls)
                denom = sum((x - xm) ** 2 for x in xs)
                m = (sum((x - xm) * (l - lm) for x, l in zip(xs, Ls)) / denom) if denom else 0.0
            else:
                m = (Ls[-1] - Ls[0]) / max(hours_between(ts[0], ts[-1]), 1.0 / 6.0)
        else:
            m = 0.0
        m = max(-1.4, min(1.4, m))
        age_h = max(0.0, age_h)
        coins.append({
            "mint": mint, "L": L, "m": m, "pool": pool,
            "surv": survival_p(L, pool, age_h),
            "sig": SIGMA_BASE if pool else SIGMA_YOUNG,
        })
    # keep only coins plausibly able to reach the top fight (cap floor + a margin)
    coins = [c for c in coins if c["L"] >= CONTEND - 1.0]

    # ---- Monte Carlo over the horizon
    # per-read top-5 counts and size samples
    place_count = {k: {} for k in range(1, places + 1)}          # mint -> count
    place_bins = {k: [0] * nbins for k in range(1, places + 1)}  # bin count
    place_mom = {k: [0.0, 0.0, 0.0] for k in range(1, places + 1)}  # n, sum, sum2 of log-cap
    for s in range(N_SCEN):
        for c in coins:
            if rng.random() < c["surv"]:
                cp, hp, fp = regime_probs(c["m"])
                u = rng.random()
                if u < cp:
                    drift = CLIMB_FLOOR + CLIMB_MOM * min(c["m"], 0.9)
                elif u < cp + hp:
                    drift = HOLD_DRIFT
                else:
                    drift = FADE_DRIFT
            else:
                drift = None  # collapsed
            c["_drift"] = drift
        # price each read
        for rd in reads:
            dt = hours_between(T_last, rd)
            caps = []
            for c in coins:
                if c["_drift"] is None:
                    continue
                Lf = c["L"] + c["_drift"] * dt + c["sig"] * math.sqrt(dt) * rng.gauss(0.0, 1.0)
                if Lf >= CONTEND:
                    caps.append((Lf, c))
            caps.sort(reverse=True, key=lambda x: x[0])
            for k in range(1, places + 1):
                if len(caps) >= k:
                    mint = caps[k - 1][1]["mint"]
                    Lf = caps[k - 1][0]
                    pc = place_count[k]
                    pc[mint] = pc.get(mint, 0) + 1
                    bin = int(min(max(math.floor((Lf - edges[0]) / 0.05), 0), nbins - 1))
                    if Lf < edges[0]:
                        bin = 0
                    elif Lf >= edges[-1]:
                        bin = nbins - 1
                    place_bins[k][bin] += 1
                    mom = place_mom[k]
                    mom[0] += 1.0
                    mom[1] += Lf
                    mom[2] += Lf * Lf

    # horizon size for sizing / newcomer hedge
    avg_dt = sum(hours_between(T_last, rd) for rd in reads) / len(reads)

    # ---- aggregate probabilities per place
    probabilities = []
    for k in range(1, places + 1):
        pc = place_count[k]
        total = N_SCEN * len(reads)
        if not pc:
            continue
        # probability mass distribution among named coins
        mass = [(mint, cnt / total) for mint, cnt in pc.items()]
        mass.sort(key=lambda x: x[1], reverse=True)
        s = sum(p for _, p in mass)
        # cap the mass on known coins at TARGET_NAMED, leaving the rest unnamed
        # for the coins born after the cut (unknowable) and the far fades.
        target = min(TARGET_NAMED[k], s)
        scale = target / s if s > 0 else 0.0
        for mint, p in mass:
            p = p * scale
            if p < EMIT_FLOOR:
                continue
            p = math.floor(p * MICRO) / MICRO
            if p >= MIN_P:
                probabilities.append({"mint": mint, "place": k, "p": p})

    # ---- size bins per place: a horizon-aware normal at the place's expected cap.
    # The cap moves by roughly sqrt(horizon): tight near the cut, wide by the far
    # reads. The expected place-k cap is anchored to the coin currently in kth
    # place (the k-th market cap at the cut), drifting only slightly.
    avg_dt_h = max(avg_dt, 0.05)
    size_sd = min(0.95, max(0.30, 0.32 * math.sqrt(avg_dt_h)))
    size_drift = -0.012
    sizes = []
    for k in range(1, places + 1):
        if k <= len(last_rows):
            mu = size_of(last_rows[k - 1]["market_cap_lamports"]) + size_drift * avg_dt_h
        else:
            mu = math.log10(1800.0) + size_drift * avg_dt_h
        arr = [norm_area(-math.inf if bi == 0 else edges[bi - 1],
                         math.inf if bi == nbins - 1 else edges[bi], mu, size_sd)
               for bi in range(nbins)]
        # a monster newcomer can land a much larger cap than any known coin:
        # hedge a share of each place's size on the top bins.
        tail = NEWCOMER_TAIL
        extra = tail / 8.0
        for bi in range(max(0, nbins - 8), nbins):
            arr[bi] += extra
        s = sum(arr)
        scale = (1.0 - SIZE_UNNAMED) / s if s > 0 else 0.0
        for bi in range(nbins):
            p = arr[bi] * scale
            p = math.floor(p * MICRO) / MICRO
            if p >= MIN_P:
                sizes.append({"place": k, "bin": bi, "p": p})

    # ---- top-5 greedy from marginals (distinct), falling back to overall counts
    allcnt = {}
    for pc2 in place_count.values():
        for mint, cnt in pc2.items():
            allcnt[mint] = allcnt.get(mint, 0) + cnt
    top5 = []
    taken = set()
    for k in range(1, places + 1):
        pc = place_count[k]
        ranked = sorted(((mint, cnt) for mint, cnt in pc.items() if mint not in taken),
                        key=lambda x: x[1], reverse=True)
        if ranked:
            top5.append(ranked[0][0])
            taken.add(ranked[0][0])
        else:
            rem = sorted(((mint, cnt) for mint, cnt in allcnt.items() if mint not in taken),
                         key=lambda x: x[1], reverse=True)
            if rem:
                top5.append(rem[0][0])
                taken.add(rem[0][0])

    forecast = {
        "probabilities": probabilities,
        "top5": top5[:places],
        "sizes": sizes,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task.get("output", "forecast.json")), "w") as f:
        f.write(json.dumps(forecast, separators=(",", ":")) + "\n")


def norm_area(lo, hi, mu, sd):
    return 0.5 * (erfc((lo - mu) / (sd * math.sqrt(2.0))) - erfc((hi - mu) / (sd * math.sqrt(2.0))))


def erfc(x):
    z = abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    r = t * math.exp(-z * z - 1.26551223 +
                    t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
                    t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return r if x >= 0 else 2.0 - r


if __name__ == "__main__":
    main()