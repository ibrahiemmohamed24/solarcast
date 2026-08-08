"""
SolarCast - Grid collision detection.

NASA POWER serves gridded data. Sites closer together than the grid
resolution return identical or near-identical series. Ignoring this
inflates apparent sample size and leaks information between train and
test splits when co-located sites land on opposite sides of a split.

Detection is empirical, not geometric: we compare the actual series.
Geometric proximity alone is misleading because NASA POWER does not use
one resolution for all variables. Irradiance comes from a finer
satellite grid than temperature and wind, so two sites can share a
temperature cell while having distinct irradiance.

Three outcomes:
  duplicate       all variables identical, drop from training
  partial_overlap some variables identical, must stay in same split
  independent     safe to treat separately
"""

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

VALUE_COLS = [
    "ALLSKY_SFC_SW_DWN",
    "CLRSKY_SFC_SW_DWN",
    "T2M",
    "WS10M",
    "WS50M",
    "WD10M",
    "RH2M",
    "PS",
]

IDENTICAL_TOL = 1e-6

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SILVER_DIR = PROJECT_ROOT / "data" / "silver" / "weather_hourly"
OUT_PATH = PROJECT_ROOT / "config" / "grid_collisions.json"
LOG_DIR = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "collisions.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("grid_collisions")


def load_silver():
    files = sorted(SILVER_DIR.glob("site_id=*/year=*/part.parquet"))
    if not files:
        raise FileNotFoundError(f"No silver files under {SILVER_DIR}")
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def series_by_site(df):
    out = {}
    for site_id, group in df.groupby("site_id"):
        out[site_id] = group.sort_values("timestamp_utc")[VALUE_COLS].reset_index(drop=True)
    return out


def full_hash(frame):
    return hashlib.md5(frame.round(4).to_numpy().tobytes()).hexdigest()[:12]


def shared_variables(a, b):
    shared = []
    for col in VALUE_COLS:
        if (a[col] - b[col]).abs().max() < IDENTICAL_TOL:
            shared.append(col)
    return shared


def detect(df):
    series = series_by_site(df)
    site_ids = sorted(series)

    hashes = {sid: full_hash(frame) for sid, frame in series.items()}

    pairs = []
    for i, a in enumerate(site_ids):
        for b in site_ids[i + 1:]:
            shared = shared_variables(series[a], series[b])
            if not shared:
                continue
            kind = "duplicate" if len(shared) == len(VALUE_COLS) else "partial_overlap"
            pairs.append({
                "site_a": a,
                "site_b": b,
                "relation": kind,
                "shared_variables": shared,
                "n_shared": len(shared),
            })

    groups = {}
    for sid in site_ids:
        groups.setdefault(hashes[sid], []).append(sid)
    duplicate_groups = [g for g in groups.values() if len(g) > 1]

    linked = {}
    for pair in pairs:
        linked.setdefault(pair["site_a"], set()).add(pair["site_b"])
        linked.setdefault(pair["site_b"], set()).add(pair["site_a"])

    split_groups = []
    seen = set()
    for sid in site_ids:
        if sid in seen:
            continue
        stack, cluster = [sid], []
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            cluster.append(cur)
            stack.extend(linked.get(cur, []))
        split_groups.append(sorted(cluster))

    keep, drop = [], []
    for group in duplicate_groups:
        keep.append(sorted(group)[0])
        drop.extend(sorted(group)[1:])
    for sid in site_ids:
        if sid not in drop and sid not in keep:
            keep.append(sid)

    return {
        "n_sites": len(site_ids),
        "n_unique_series": len(set(hashes.values())),
        "duplicate_groups": duplicate_groups,
        "pairs": pairs,
        "split_groups": split_groups,
        "training_sites": sorted(keep),
        "excluded_duplicates": sorted(drop),
    }


def run():
    df = load_silver()
    report = detect(df)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    log.info("sites=%d unique_series=%d", report["n_sites"], report["n_unique_series"])
    log.info("duplicate groups: %s", report["duplicate_groups"])
    log.info("excluded as duplicates: %s", report["excluded_duplicates"])
    for pair in report["pairs"]:
        if pair["relation"] == "partial_overlap":
            log.warning(
                "%s and %s share %d/%d variables: %s",
                pair["site_a"], pair["site_b"],
                pair["n_shared"], len(VALUE_COLS), pair["shared_variables"],
            )
    log.info("split groups that must stay together: %s",
             [g for g in report["split_groups"] if len(g) > 1])
    log.info("Report written to %s", OUT_PATH)

    return report


if __name__ == "__main__":
    run()
