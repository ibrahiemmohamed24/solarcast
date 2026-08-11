"""
Null tests for the wind model.

The model lands within 5 percent of reported generation at Zafarana and
Gabal El Zeit. That only means something if a model fed broken inputs
does noticeably worse. These tests break one input at a time and rerun:
shuffle the wind series in time, swap wind between sites, and replace it
with random draws that keep only the mean.

Thresholds were fixed before running: under 15 percent error on a broken
input means the original match is weak evidence, over 30 percent means
the model is genuinely using the data.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models.calibrate_wind import (
    load_config, load_site, site_physics, simulate_fleet, net_factor
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "config" / "permutation_tests.json"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "permutation_tests.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("permutation")

SEED = 42
N_REPEATS = 20
WEAK_THRESHOLD = 15.0
STRONG_THRESHOLD = 30.0

WIND_COLS = ["WS10M", "WS50M"]


def run_site(df, cfg, site_id):
    farm = cfg["farm_fleets"][site_id]
    net = net_factor(cfg["loss_factors"])
    alpha, alpha_med, rho = site_physics(df)
    years = df.year.nunique()
    total_kwh, _ = simulate_fleet(df, alpha, rho, farm["fleet"], cfg["turbines"])
    return float(total_kwh.sum() / 1e6 / years) * net


def error_pct(modelled, reported):
    return (modelled - reported) / reported * 100


def shuffle_time(df, rng):
    out = df.copy()
    order = rng.permutation(len(out))
    for col in WIND_COLS:
        out[col] = out[col].values[order]
    return out


def swap_sites(df_a, df_b):
    out = df_a.copy()
    n = min(len(out), len(df_b))
    out = out.iloc[:n].copy()
    for col in WIND_COLS:
        out[col] = df_b[col].values[:n]
    return out


def random_same_mean(df, rng):
    out = df.copy()
    for col in WIND_COLS:
        values = out[col].values
        mean = np.nanmean(values)
        shape = 2.0
        scale = mean / 0.8862
        out[col] = rng.weibull(shape, len(values)) * scale
    return out


def run():
    cfg = load_config()
    rng = np.random.default_rng(SEED)

    sites = [s for s, f in cfg["farm_fleets"].items()
             if f.get("reported_generation_gwh")]
    log.info("Sites with reported generation: %s", sites)

    data = {s: load_site(s) for s in sites}
    results = []

    for site_id in sites:
        df = data[site_id]
        reported = cfg["farm_fleets"][site_id]["reported_generation_gwh"]
        baseline = run_site(df, cfg, site_id)
        base_err = error_pct(baseline, reported)

        log.info("%s | baseline %.0f GWh vs reported %.0f (%+.1f%%)",
                 site_id, baseline, reported, base_err)

        shuffled = [error_pct(run_site(shuffle_time(df, rng), cfg, site_id), reported)
                    for _ in range(N_REPEATS)]
        randomised = [error_pct(run_site(random_same_mean(df, rng), cfg, site_id), reported)
                      for _ in range(N_REPEATS)]

        swaps = {}
        for other in sites:
            if other == site_id:
                continue
            swapped = swap_sites(df, data[other])
            swaps[other] = error_pct(run_site(swapped, cfg, site_id), reported)

        log.info("  time shuffle : %+.1f%% (sd %.1f)",
                 np.mean(shuffled), np.std(shuffled))
        log.info("  random wind  : %+.1f%% (sd %.1f)",
                 np.mean(randomised), np.std(randomised))
        for other, err in swaps.items():
            log.info("  wind from %-18s %+.1f%%", other, err)

        results.append({
            "site_id": site_id,
            "reported_gwh": reported,
            "baseline_gwh": baseline,
            "baseline_error_pct": base_err,
            "time_shuffle_mean_error_pct": float(np.mean(shuffled)),
            "time_shuffle_sd": float(np.std(shuffled)),
            "random_wind_mean_error_pct": float(np.mean(randomised)),
            "random_wind_sd": float(np.std(randomised)),
            "site_swap_errors": swaps,
        })

    log.info("")
    log.info("%-18s %8s %10s %10s %10s", "site", "base", "shuffled", "random", "swap_worst")
    for r in results:
        worst = max(abs(v) for v in r["site_swap_errors"].values()) if r["site_swap_errors"] else np.nan
        log.info("%-18s %+7.1f%% %+9.1f%% %+9.1f%% %9.1f%%",
                 r["site_id"], r["baseline_error_pct"],
                 r["time_shuffle_mean_error_pct"],
                 r["random_wind_mean_error_pct"], worst)

    degradations = []
    for r in results:
        for key in ("time_shuffle_mean_error_pct", "random_wind_mean_error_pct"):
            degradations.append(abs(r[key]))
        degradations.extend(abs(v) for v in r["site_swap_errors"].values())

    median_broken = float(np.median(degradations))
    log.info("")
    log.info("Median absolute error across all broken-input runs: %.1f%%", median_broken)
    if median_broken < WEAK_THRESHOLD:
        verdict = "weak_evidence"
        log.warning("Below %.0f%%: matching reported generation is weak evidence",
                    WEAK_THRESHOLD)
    elif median_broken > STRONG_THRESHOLD:
        verdict = "model_uses_data"
        log.info("Above %.0f%%: the model is genuinely using the wind data",
                 STRONG_THRESHOLD)
    else:
        verdict = "inconclusive"
        log.warning("Between thresholds: inconclusive")

    report = {
        "protocol": {
            "n_repeats": N_REPEATS,
            "seed": SEED,
            "weak_threshold_pct": WEAK_THRESHOLD,
            "strong_threshold_pct": STRONG_THRESHOLD,
            "fixed_before_running": True,
        },
        "results": results,
        "median_broken_error_pct": median_broken,
        "verdict": verdict,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("Written to %s", OUT_PATH)
    return report


if __name__ == "__main__":
    run()
