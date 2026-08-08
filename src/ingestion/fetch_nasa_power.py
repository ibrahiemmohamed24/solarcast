"""
SolarCast - NASA POWER hourly weather ingestion.

Pulls raw hourly weather data for every site in config/sites.yaml
and writes untouched JSON responses to the bronze layer.
Raw stays raw: no cleaning, no type casting, no filtering here.
"""

import json
import logging
import time
from pathlib import Path

import requests
import yaml

BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

PARAMETERS = [
    "ALLSKY_SFC_SW_DWN",
    "CLRSKY_SFC_SW_DWN",
    "T2M",
    "WS10M",
    "WS50M",
    "WD10M",
    "RH2M",
    "PS",
]

REQUEST_DELAY_SEC = 2
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 10
TIMEOUT_SEC = 120

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "sites.yaml"
BRONZE_DIR = PROJECT_ROOT / "data" / "bronze" / "nasa_power"
LOG_DIR = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)
BRONZE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "ingestion.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("nasa_power")


def load_sites(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    sites = list(config.get("egypt_sites", []))
    sites.extend(config.get("validation_zones", []))
    return sites


def build_params(latitude, longitude, start, end):
    return {
        "parameters": ",".join(PARAMETERS),
        "community": "RE",
        "latitude": latitude,
        "longitude": longitude,
        "start": start,
        "end": end,
        "format": "JSON",
    }


def fetch_one(site_id, latitude, longitude, year):
    start = f"{year}0101"
    end = f"{year}1231"
    params = build_params(latitude, longitude, start, end)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(BASE_URL, params=params, timeout=TIMEOUT_SEC)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            log.warning(
                "%s %s attempt %d/%d failed: %s",
                site_id, year, attempt, MAX_RETRIES, exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * attempt)

    log.error("%s %s gave up after %d attempts", site_id, year, MAX_RETRIES)
    return None


def save_raw(payload, site_id, year):
    site_dir = BRONZE_DIR / f"site_id={site_id}"
    site_dir.mkdir(parents=True, exist_ok=True)
    out_path = site_dir / f"year={year}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return out_path


def run(start_year, end_year, overwrite=False):
    sites = load_sites()
    log.info("Loaded %d sites from config", len(sites))

    succeeded = 0
    skipped = 0
    failed = []

    for site in sites:
        site_id = site["id"]
        latitude = site["latitude"]
        longitude = site["longitude"]

        for year in range(start_year, end_year + 1):
            out_path = BRONZE_DIR / f"site_id={site_id}" / f"year={year}.json"

            if out_path.exists() and not overwrite:
                log.info("%s %s already present, skipping", site_id, year)
                skipped += 1
                continue

            log.info("Fetching %s %s", site_id, year)
            payload = fetch_one(site_id, latitude, longitude, year)

            if payload is None:
                failed.append((site_id, year))
            else:
                save_raw(payload, site_id, year)
                succeeded += 1

            time.sleep(REQUEST_DELAY_SEC)

    log.info("Done. ok=%d skipped=%d failed=%d", succeeded, skipped, len(failed))
    if failed:
        log.warning("Failed pairs: %s", failed)

    return {"ok": succeeded, "skipped": skipped, "failed": failed}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch NASA POWER hourly data")
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run(args.start_year, args.end_year, args.overwrite)
