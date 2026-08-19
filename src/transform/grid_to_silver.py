"""
Turns the raw grid JSON into partitioned Parquet.

Same flattening as the site pipeline, but written for 11 GB: files are
processed one at a time and never all held in memory. Years are grouped
into one file per grid point, because thousands of tiny files slow Spark
down and multiply object-store requests.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

MISSING_SENTINEL = -999.0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRID_BRONZE = PROJECT_ROOT / "data" / "bronze" / "nasa_power_grid"
GRID_SILVER = PROJECT_ROOT / "data" / "silver" / "grid_hourly"
GRID_CONFIG = PROJECT_ROOT / "config" / "grid_points.json"
LOG_DIR = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)
GRID_SILVER.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "grid_transform.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("grid_to_silver")

VALUE_COLS = [
    "ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "T2M",
    "WS10M", "WS50M", "WD10M", "RH2M", "PS",
]


def load_points():
    with open(GRID_CONFIG, "r", encoding="utf-8") as f:
        return {p["id"]: p for p in json.load(f)["grid_points"]}


def flatten(path, point):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    frame = pd.DataFrame(payload["properties"]["parameter"])
    frame.index.name = "ts_raw"
    frame = frame.reset_index()

    for col in VALUE_COLS:
        if col not in frame.columns:
            frame[col] = np.nan
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    sentinels = int((frame[VALUE_COLS] == MISSING_SENTINEL).sum().sum())
    frame[VALUE_COLS] = frame[VALUE_COLS].replace(MISSING_SENTINEL, np.nan)

    frame["timestamp_utc"] = pd.to_datetime(frame.ts_raw, format="%Y%m%d%H", utc=True)
    frame = frame.drop(columns=["ts_raw"])

    frame["point_id"] = point["id"]
    frame["latitude"] = np.float32(point["latitude"])
    frame["longitude"] = np.float32(point["longitude"])
    frame["year"] = frame.timestamp_utc.dt.year.astype("int16")

    for col in VALUE_COLS:
        frame[col] = frame[col].astype("float32")

    ordered = ["point_id", "timestamp_utc", "year", "latitude", "longitude"] + VALUE_COLS
    return frame[ordered], sentinels


def run(overwrite=False):
    points = load_points()
    log.info("Grid points in config: %d", len(points))

    total_rows = 0
    total_sentinels = 0
    written = 0
    skipped = 0
    failed = []

    for i, (point_id, point) in enumerate(sorted(points.items()), start=1):
        out_dir = GRID_SILVER / f"point_id={point_id}"
        out_path = out_dir / "part.parquet"

        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        files = sorted((GRID_BRONZE / f"site_id={point_id}").glob("year=*.json"))
        if not files:
            failed.append((point_id, "no bronze files"))
            continue

        frames = []
        sentinels = 0
        try:
            for path in files:
                frame, n = flatten(path, point)
                frames.append(frame)
                sentinels += n
        except Exception as exc:
            failed.append((point_id, str(exc)))
            continue

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values("timestamp_utc").reset_index(drop=True)

        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = out_dir / "part.parquet.tmp"
        combined.to_parquet(tmp_path, index=False, compression="snappy")
        tmp_path.replace(out_path)

        written += 1
        total_rows += len(combined)
        total_sentinels += sentinels

        if sentinels:
            log.warning("%s had %d sentinel values", point_id, sentinels)
        if i % 50 == 0:
            log.info("%d/%d points | rows so far %d", i, len(points), total_rows)

    log.info("Done. written=%d skipped=%d failed=%d rows=%d sentinels=%d",
             written, skipped, len(failed), total_rows, total_sentinels)
    if failed:
        log.warning("Failures: %s", failed[:10])

    return {"written": written, "rows": total_rows, "failed": failed}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.overwrite)
