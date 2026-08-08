"""
SolarCast - Bronze to Silver transformation.

Flattens nested NASA POWER JSON into tidy hourly rows, handles
sentinel missing values, attaches site metadata, and writes
partitioned Parquet.

One row = one hour at one site.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import yaml

MISSING_SENTINEL = -999.0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "sites.yaml"
BRONZE_DIR = PROJECT_ROOT / "data" / "bronze" / "nasa_power"
SILVER_DIR = PROJECT_ROOT / "data" / "silver" / "weather_hourly"
LOG_DIR = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)
SILVER_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "transform.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("bronze_to_silver")

LOCAL_TZ = {
    "EG": "Africa/Cairo",
    "ES": "Europe/Madrid",
    "DE": "Europe/Berlin",
    "PT": "Europe/Lisbon",
    "IT": "Europe/Rome",
}


def load_site_lookup():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    lookup = {}

    for site in config.get("egypt_sites", []):
        lookup[site["id"]] = {
            "site_name": site["name"],
            "country": "EG",
            "region": site.get("governorate"),
            "tech": site.get("tech"),
            "capacity_mw": site.get("capacity_mw"),
            "status": site.get("status"),
            "data_status": site.get("data_status"),
            "latitude": site["latitude"],
            "longitude": site["longitude"],
        }

    for zone in config.get("validation_zones", []):
        lookup[zone["id"]] = {
            "site_name": zone["name"],
            "country": zone["id"],
            "region": None,
            "tech": "mixed",
            "capacity_mw": None,
            "status": "validation_zone",
            "data_status": zone.get("data_status"),
            "latitude": zone["latitude"],
            "longitude": zone["longitude"],
        }

    return lookup


def flatten_payload(payload, site_id):
    params = payload["properties"]["parameter"]
    frame = pd.DataFrame(params)
    frame.index.name = "ts_raw"
    frame = frame.reset_index()
    frame.insert(0, "site_id", site_id)
    return frame


def clean_frame(frame, site_meta):
    value_cols = [c for c in frame.columns if c not in ("site_id", "ts_raw")]

    for col in value_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    missing_before = int((frame[value_cols] == MISSING_SENTINEL).sum().sum())
    frame[value_cols] = frame[value_cols].replace(MISSING_SENTINEL, pd.NA)

    frame["timestamp_utc"] = pd.to_datetime(
        frame["ts_raw"], format="%Y%m%d%H", utc=True
    )
    frame = frame.drop(columns=["ts_raw"])

    tz = LOCAL_TZ.get(site_meta["country"], "UTC")
    frame["timestamp_local"] = frame["timestamp_utc"].dt.tz_convert(tz)

    for key, value in site_meta.items():
        frame[key] = value

    frame["year"] = frame["timestamp_utc"].dt.year

    ordered = (
        ["site_id", "site_name", "country", "timestamp_utc", "timestamp_local"]
        + value_cols
        + ["latitude", "longitude", "region", "tech", "capacity_mw",
           "status", "data_status", "year"]
    )
    frame = frame[ordered]

    return frame, missing_before


def process_file(path, lookup):
    site_id = path.parent.name.split("=")[1]
    year = path.stem.split("=")[1]

    if site_id not in lookup:
        log.warning("Unknown site %s, skipping", site_id)
        return None

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    frame = flatten_payload(payload, site_id)
    frame, missing = clean_frame(frame, lookup[site_id])

    out_dir = SILVER_DIR / f"site_id={site_id}" / f"year={year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "part.parquet"
    frame.to_parquet(out_path, index=False, compression="snappy")

    return {
        "site_id": site_id,
        "year": year,
        "rows": len(frame),
        "missing": missing,
        "path": out_path,
    }


def run():
    lookup = load_site_lookup()
    files = sorted(BRONZE_DIR.glob("site_id=*/year=*.json"))
    log.info("Found %d bronze files", len(files))

    total_rows = 0
    total_missing = 0
    processed = 0

    for path in files:
        try:
            result = process_file(path, lookup)
        except Exception as exc:
            log.error("Failed on %s: %s", path, exc)
            continue

        if result is None:
            continue

        processed += 1
        total_rows += result["rows"]
        total_missing += result["missing"]

        if result["missing"] > 0:
            log.warning(
                "%s %s had %d sentinel values",
                result["site_id"], result["year"], result["missing"],
            )

    log.info(
        "Done. files=%d rows=%d sentinels_replaced=%d",
        processed, total_rows, total_missing,
    )
    return {"files": processed, "rows": total_rows, "missing": total_missing}


if __name__ == "__main__":
    run()
