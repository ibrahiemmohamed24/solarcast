"""
Decides how to pick reference curves: average the whole library, or
average the five nearest by rated-to-cut-in ratio and specific power.

Protocol was fixed before running. Primary evaluation holds out the whole
turbine family, because none of the Egyptian turbines have a sibling in
the library and the leave-one-out number flatters nearest-neighbour
selection. Decision rule: adopt ratio_sp only if the lower bound of a
family-level paired bootstrap on the median difference clears 0.3
percentage points.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.power_curves import (
    build_reference_index,
    load_reference_curve,
    curve_to_cp,
    AIR_DENSITY_STANDARD,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "config" / "reference_selection.json"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "reference_selection.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("ref_selection")

V = np.arange(0.5, 30.01, 0.5)
N_REFS = 5
PRACTICAL_MARGIN_PP = 0.3
N_BOOTSTRAP = 5000
SEED = 42

WEIBULL_GULF = (11.5, 3.3)
WEIBULL_MILD = (7.0, 2.0)


def weibull_weights(scale, shape):
    pdf = (shape / scale) * (V / scale) ** (shape - 1) * np.exp(-((V / scale) ** shape))
    return pdf / pdf.sum()


W_GULF = weibull_weights(*WEIBULL_GULF)
W_MILD = weibull_weights(*WEIBULL_MILD)


def curve_on_grid(turbine_type):
    curve = load_reference_curve(turbine_type)
    return np.interp(V, curve.wind_speed_ms.values, curve.power_kw.values,
                     left=0, right=0)


def operating_points(turbine_type):
    curve = load_reference_curve(turbine_type)
    rated = curve.power_kw.max()
    cut_in = float(curve.loc[curve.power_kw > 0, "wind_speed_ms"].iloc[0])
    rated_ms = float(
        curve.loc[curve.power_kw >= 0.99 * rated, "wind_speed_ms"].iloc[0]
    )
    return cut_in, rated_ms, rated


def prepare_index():
    index = build_reference_index()
    index["family"] = index.turbine_type.str.split("/").str[0]
    points = [operating_points(t) for t in index.turbine_type]
    index["cut_in"] = [p[0] for p in points]
    index["rated_ms"] = [p[1] for p in points]
    index["ratio"] = index.rated_ms / index.cut_in
    return index


def rebuild(target, references):
    t_cut_in, t_rated_ms, t_max = operating_points(target.turbine_type)
    area = np.pi * (target.diameter_m / 2) ** 2

    stack = []
    for ref_type, ref_diameter in references:
        r_cut_in, r_rated_ms, _ = operating_points(ref_type)
        ref_cp = curve_to_cp(load_reference_curve(ref_type), ref_diameter)
        scale = (t_rated_ms - t_cut_in) / (r_rated_ms - r_cut_in)
        equivalent = r_cut_in + (V - t_cut_in) / scale
        stack.append(np.interp(equivalent, ref_cp.wind_speed_ms.values,
                               ref_cp.cp.values, left=0, right=0))

    cp = np.mean(np.vstack(stack), axis=0)
    power = cp * 0.5 * AIR_DENSITY_STANDARD * area * V ** 3 / 1000
    power = np.where(V < t_cut_in, 0, power)
    power = np.where(V >= t_rated_ms, t_max, power)
    return np.clip(power, 0, t_max)


def aep_error_pct(estimated, actual, weights):
    actual_energy = (actual * weights).sum()
    if actual_energy <= 0:
        return np.nan
    return ((estimated * weights).sum() - actual_energy) / actual_energy * 100


def select_ratio_sp(pool, target, n_refs=N_REFS):
    score = (
        ((pool.ratio - target.ratio) / pool.ratio.std()) ** 2
        + ((pool.specific_power - target.specific_power) / pool.specific_power.std()) ** 2
    )
    chosen = pool.assign(_score=score).nsmallest(n_refs, "_score")
    return list(zip(chosen.turbine_type, chosen.diameter_m)), chosen


def evaluate(index, holdout="family"):
    rows = []
    for _, target in index.iterrows():
        if holdout == "family":
            pool = index[index.family != target.family]
        else:
            pool = index[index.turbine_type != target.turbine_type]

        if len(pool) < N_REFS:
            continue

        actual = curve_on_grid(target.turbine_type)

        all_refs = list(zip(pool.turbine_type, pool.diameter_m))
        est_all = rebuild(target, all_refs)

        sel_refs, chosen = select_ratio_sp(pool, target)
        est_sel = rebuild(target, sel_refs)

        nearest = chosen.iloc[0]
        exact_sp_matches = int(
            (pool.specific_power.round() == round(target.specific_power)).sum()
        )

        rows.append({
            "turbine_type": target.turbine_type,
            "family": target.family,
            "diameter_m": target.diameter_m,
            "specific_power": round(target.specific_power, 1),
            "ratio": round(target.ratio, 2),
            "err_all_gulf": aep_error_pct(est_all, actual, W_GULF),
            "err_sel_gulf": aep_error_pct(est_sel, actual, W_GULF),
            "err_all_mild": aep_error_pct(est_all, actual, W_MILD),
            "err_sel_mild": aep_error_pct(est_sel, actual, W_MILD),
            "nearest_ref": nearest.turbine_type,
            "nearest_ref_d_sp": round(nearest.specific_power - target.specific_power, 1),
            "nearest_ref_d_ratio": round(nearest.ratio - target.ratio, 2),
            "n_exact_sp_matches": exact_sp_matches,
        })

    return pd.DataFrame(rows)


def paired_bootstrap(results, rng):
    families = results.family.unique()
    deltas = []

    for _ in range(N_BOOTSTRAP):
        drawn = rng.choice(families, size=len(families), replace=True)
        sample = pd.concat([results[results.family == f] for f in drawn])
        median_all = sample.err_all_gulf.abs().median()
        median_sel = sample.err_sel_gulf.abs().median()
        deltas.append(median_all - median_sel)

    deltas = np.array(deltas)
    return {
        "delta_point": float(
            results.err_all_gulf.abs().median() - results.err_sel_gulf.abs().median()
        ),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_delta_gt_margin": float((deltas > PRACTICAL_MARGIN_PP).mean()),
    }


def run():
    index = prepare_index()
    log.info("Library: %d turbines in %d families",
             len(index), index.family.nunique())

    primary = evaluate(index, holdout="family")
    secondary = evaluate(index, holdout="turbine")

    log.info("PRIMARY (leave-one-family-out), n=%d", len(primary))
    for label, col in [("average all", "err_all_gulf"), ("ratio_sp n=5", "err_sel_gulf")]:
        s = primary[col].abs()
        log.info("  %-14s gulf: median %.2f  mean %.2f  p90 %.2f",
                 label, s.median(), s.mean(), s.quantile(0.9))
    for label, col in [("average all", "err_all_mild"), ("ratio_sp n=5", "err_sel_mild")]:
        s = primary[col].abs()
        log.info("  %-14s mild: median %.2f  mean %.2f  p90 %.2f",
                 label, s.median(), s.mean(), s.quantile(0.9))

    log.info("SECONDARY (leave-one-turbine-out, sibling available), n=%d", len(secondary))
    for label, col in [("average all", "err_all_gulf"), ("ratio_sp n=5", "err_sel_gulf")]:
        s = secondary[col].abs()
        log.info("  %-14s gulf: median %.2f  p90 %.2f", label, s.median(), s.quantile(0.9))

    rng = np.random.default_rng(SEED)
    boot = paired_bootstrap(primary, rng)

    log.info("Paired family-level bootstrap on median difference:")
    log.info("  delta = %.3f pp  (95%% CI %.3f to %.3f)",
             boot["delta_point"], boot["ci_low"], boot["ci_high"])
    log.info("  P(delta > %.1f pp) = %.3f", PRACTICAL_MARGIN_PP, boot["p_delta_gt_margin"])

    adopt = boot["ci_low"] > PRACTICAL_MARGIN_PP
    decision = "ratio_sp_n5" if adopt else "average_all"
    log.info("DECISION: %s", decision)
    if not adopt:
        log.info("  ratio_sp not adopted: lower CI bound does not clear the margin")

    diag = primary[["turbine_type", "family", "nearest_ref",
                    "nearest_ref_d_sp", "nearest_ref_d_ratio",
                    "n_exact_sp_matches"]]
    log.info("Cross-manufacturer near-duplicates (same rounded specific power): %d families affected",
             int((diag.n_exact_sp_matches > 0).sum()))

    report = {
        "protocol": {
            "primary": "leave_one_family_out",
            "secondary": "leave_one_turbine_out",
            "metric": "absolute AEP error, Weibull A=11.5 k=3.3",
            "n_refs": N_REFS,
            "practical_margin_pp": PRACTICAL_MARGIN_PP,
            "bootstrap_draws": N_BOOTSTRAP,
            "seed": SEED,
            "fixed_before_running": True,
        },
        "bootstrap": boot,
        "decision": decision,
        "primary_summary": {
            "average_all": {
                "median": float(primary.err_all_gulf.abs().median()),
                "mean": float(primary.err_all_gulf.abs().mean()),
                "p90": float(primary.err_all_gulf.abs().quantile(0.9)),
            },
            "ratio_sp_n5": {
                "median": float(primary.err_sel_gulf.abs().median()),
                "mean": float(primary.err_sel_gulf.abs().mean()),
                "p90": float(primary.err_sel_gulf.abs().quantile(0.9)),
            },
        },
        "secondary_summary": {
            "average_all_median": float(secondary.err_all_gulf.abs().median()),
            "ratio_sp_n5_median": float(secondary.err_sel_gulf.abs().median()),
        },
        "diagnostics": diag.to_dict(orient="records"),
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("Written to %s", OUT_PATH)

    return report


if __name__ == "__main__":
    run()
