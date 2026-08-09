"""
Pulls NASA POWER hourly data for every grid point.

Same fetch logic as the site ingestion, but reads from grid_points.json
and writes to a separate bronze folder. Runs are resumable: anything
already on disk is skipped, so a dropped connection costs you one request
not the whole run.
"""

import json
import logging
import time
from pathlib import Path

from src.ingestion.fetch_nasa_power import fetch_one, save_raw, BRONZE_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRID_PATH = PROJECT_ROOT / "config" / "grid_points.json"
GRID_BRONZE = PROJECT_ROOT / "data" / "bronze" / "nasa_power_grid"
LOG_DIR = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)
GRID_BRONZE.mkdir(parents=True, exist_ok=True)

REQUEST_DELAY_SEC = 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "grid_ingestion.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("fetch_grid")


def load_grid():
    with open(GRID_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["grid_points"]


def save_grid_raw(payload, point_id, year):
    point_dir = GRID_BRONZE / f"site_id={point_id}"
    point_dir.mkdir(parents=True, exist_ok=True)
    out_path = point_dir / f"year={year}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return out_path


def pending_work(points, start_year, end_year):
    todo = []
    for point in points:
        for year in range(start_year, end_year + 1):
            path = GRID_BRONZE / f"site_id={point['id']}" / f"year={year}.json"
            if not path.exists():
                todo.append((point, year))
    return todo


def run(start_year, end_year):
    points = load_grid()
    todo = pending_work(points, start_year, end_year)
    total = len(points) * (end_year - start_year + 1)

    log.info("Grid points: %d | years: %d-%d", len(points), start_year, end_year)
    log.info("Total tasks: %d | already done: %d | pending: %d",
             total, total - len(todo), len(todo))

    if not todo:
        log.info("Nothing to do")
        return {"ok": 0, "failed": []}

    eta_min = len(todo) * REQUEST_DELAY_SEC / 60
    log.info("Rough ETA: %.0f minutes", eta_min)

    ok = 0
    failed = []
    start_time = time.time()

    for i, (point, year) in enumerate(todo, start=1):
        payload = fetch_one(point["id"], point["latitude"], point["longitude"], year)

        if payload is None:
            failed.append((point["id"], year))
        else:
            save_grid_raw(payload, point["id"], year)
            ok += 1

        if i % 50 == 0:
            elapsed = (time.time() - start_time) / 60
            rate = i / elapsed if elapsed else 0
            remaining = (len(todo) - i) / rate if rate else 0
            log.info("%d/%d | ok=%d failed=%d | %.1f min elapsed | ~%.0f min left",
                     i, len(todo), ok, len(failed), elapsed, remaining)

        time.sleep(REQUEST_DELAY_SEC)

    log.info("Done. ok=%d failed=%d", ok, len(failed))
    if failed:
        log.warning("Failed: %s", failed[:20])

    return {"ok": ok, "failed": failed}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2005)
    parser.add_argument("--end-year", type=int, default=2024)
    args = parser.parse_args()

    run(args.start_year, args.end_year)
