"""
Calibrates the wind model against reported generation.

Manufacturers don't publish curve tables and hub heights are often
missing from project documents, so instead of guessing we run the model
across the plausible options and see which one lands closest to what the
farm actually reported producing. Every run goes to MLflow.
"""

import glob
import json
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

from src.models.power_curves import (
    Turbine,
    shear_exponent,
    extrapolate_wind,
    air_density,
    density_corrected_speed,
    apply_power_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SILVER_DIR = PROJECT_ROOT / "data" / "silver" / "weather_hourly"
TURBINE_CFG = PROJECT_ROOT / "config" / "turbines.yaml"
OUT_PATH = PROJECT_ROOT / "config" / "wind_calibration.json"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
LOG_DIR = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "calibration.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("calibrate_wind")

mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT}/mlflow.db")


def load_config():
    with open(TURBINE_CFG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_site(site_id):
    files = sorted(SILVER_DIR.glob(f"site_id={site_id}/year=*/part.parquet"))
    if not files:
        raise FileNotFoundError(f"No silver data for {site_id}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def build_turbine(spec, hub_height):
    return Turbine(
        name=spec["name"],
        rated_power_kw=spec["rated_power_kw"],
        rotor_diameter_m=spec["rotor_diameter_m"],
        cut_in_ms=spec["cut_in_ms"],
        rated_ms=spec["rated_ms"],
        cut_out_ms=spec["cut_out_ms"],
        hub_height_m=hub_height,
        iec_class=spec.get("iec_class"),
        control=spec.get("control", "pitch"),
        source=spec.get("source"),
    )


def site_physics(df):
    alpha = shear_exponent(df.WS10M.values, df.WS50M.values, 10, 50)
    alpha_med = float(np.nanmedian(alpha))
    alpha = np.where(np.isnan(alpha), alpha_med, alpha)
    alpha = np.clip(alpha, 0.0, 0.6)
    rho = air_density(df.T2M.values, df.PS.values)
    return alpha, alpha_med, rho


def simulate_fleet(df, alpha, rho, fleet, turbines_cfg, hub_overrides=None):
    hub_overrides = hub_overrides or {}
    total_kwh = np.zeros(len(df))
    detail = []

    for unit in fleet:
        key = unit["turbine"]
        spec = turbines_cfg[key]
        hub = hub_overrides.get(key, spec["hub_height_m"])
        turbine = build_turbine(spec, hub)

        v_hub = extrapolate_wind(df.WS50M.values, 50, hub, alpha)
        v_corr = density_corrected_speed(v_hub, rho, control=turbine.control)
        kw = apply_power_curve(v_corr, turbine)

        unit_total = kw * unit["count"]
        total_kwh += unit_total

        detail.append({
            "turbine": key,
            "count": unit["count"],
            "hub_height_m": hub,
            "mean_speed_hub_ms": float(v_hub.mean()),
            "capacity_factor": float(kw.mean() / spec["rated_power_kw"]),
            "gwh_per_year": float(unit_total.sum() / 1e6 / df.year.nunique()),
        })

    return total_kwh, detail


def net_factor(losses):
    return (
        (1 - losses["wake_loss"])
        * losses["availability"]
        * (1 - losses["electrical_loss"])
        * (1 - losses["soiling_and_icing"])
    )


def calibrate_site(site_id, cfg, experiment="wind_calibration"):
    farm = cfg["farm_fleets"][site_id]
    turbines_cfg = cfg["turbines"]
    losses = cfg["loss_factors"]
    net = net_factor(losses)

    reported = farm.get("reported_generation_gwh")
    df = load_site(site_id)
    alpha, alpha_med, rho = site_physics(df)
    years = df.year.nunique()

    log.info("%s | rows=%d years=%d alpha_med=%.3f rho_mean=%.3f",
             site_id, len(df), years, alpha_med, rho.mean())

    single_type = len(farm["fleet"]) == 1
    if single_type:
        key = farm["fleet"][0]["turbine"]
        options = turbines_cfg[key].get(
            "hub_height_options", [turbines_cfg[key]["hub_height_m"]]
        )
        candidates = [{key: h} for h in options]
    else:
        candidates = [{}]

    mlflow.set_experiment(experiment)
    results = []

    for overrides in candidates:
        label = "_".join(f"{k}={v}" for k, v in overrides.items()) or "default"

        with mlflow.start_run(run_name=f"{site_id}|{label}"):
            total_kwh, detail = simulate_fleet(
                df, alpha, rho, farm["fleet"], turbines_cfg, overrides
            )
            gross_gwh = float(total_kwh.sum() / 1e6 / years)
            net_gwh = gross_gwh * net
            capacity_mw = farm["total_capacity_mw"]
            cf_net = net_gwh * 1e3 / (capacity_mw * 8760)

            mlflow.log_params({
                "site_id": site_id,
                "capacity_mw": capacity_mw,
                "n_turbine_types": len(farm["fleet"]),
                "hub_overrides": json.dumps(overrides),
                "years": years,
                "wake_loss": losses["wake_loss"],
                "availability": losses["availability"],
            })
            mlflow.log_metrics({
                "alpha_median": alpha_med,
                "air_density_mean": float(rho.mean()),
                "gross_gwh_per_year": gross_gwh,
                "net_gwh_per_year": net_gwh,
                "capacity_factor_net": cf_net,
            })

            if reported:
                error_pct = (net_gwh - reported) / reported * 100
                mlflow.log_metrics({
                    "reported_gwh": reported,
                    "error_pct": error_pct,
                    "abs_error_pct": abs(error_pct),
                })
            else:
                error_pct = None

            mlflow.log_dict({"fleet_detail": detail}, "fleet_detail.json")

            results.append({
                "site_id": site_id,
                "overrides": overrides,
                "gross_gwh": gross_gwh,
                "net_gwh": net_gwh,
                "cf_net": cf_net,
                "reported_gwh": reported,
                "error_pct": error_pct,
                "detail": detail,
            })

            log.info(
                "  %s -> net %.0f GWh/yr | CF %.3f | reported %s | err %s",
                label, net_gwh, cf_net, reported,
                f"{error_pct:+.1f}%" if error_pct is not None else "n/a",
            )

    return results


def run(sites=None):
    cfg = load_config()
    sites = sites or list(cfg["farm_fleets"])

    all_results = []
    for site_id in sites:
        try:
            all_results.extend(calibrate_site(site_id, cfg))
        except FileNotFoundError as exc:
            log.warning("Skipping %s: %s", site_id, exc)

    best = {}
    for row in all_results:
        if row["error_pct"] is None:
            continue
        sid = row["site_id"]
        if sid not in best or abs(row["error_pct"]) < abs(best[sid]["error_pct"]):
            best[sid] = row

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"all_runs": all_results, "best_per_site": best}, f, indent=2)

    log.info("Best fit per site:")
    for sid, row in best.items():
        log.info("  %s -> %s | net %.0f vs reported %.0f (%+.1f%%)",
                 sid, row["overrides"] or "default",
                 row["net_gwh"], row["reported_gwh"], row["error_pct"])

    log.info("Written to %s", OUT_PATH)
    return all_results


if __name__ == "__main__":
    run()
