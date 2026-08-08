"""
Builds a half-degree grid over Egypt and keeps only the points on land.

Filtering uses the actual Egyptian border from Natural Earth, not a
bounding box. We first tried asking NASA which points were valid, but it
returns clean data over water too, so the API can't tell land from sea.
"""

import json
import logging
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point

BORDER_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_10m_admin_0_countries.geojson"
)
COUNTRY = "Egypt"

LAT_MIN, LAT_MAX = 21.5, 32.0
LON_MIN, LON_MAX = 24.5, 37.0
STEP = 0.5

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "config" / "grid_points.json"
BORDER_CACHE = PROJECT_ROOT / "data" / "reference" / "egypt_border.geojson"
LOG_DIR = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)
BORDER_CACHE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "grid_build.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("build_grid")


def load_border():
    if BORDER_CACHE.exists():
        log.info("Using cached border")
        return gpd.read_file(BORDER_CACHE)

    log.info("Downloading border data")
    world = gpd.read_file(BORDER_URL)
    country = world[world.NAME == COUNTRY][["NAME", "geometry"]].copy()
    if country.empty:
        raise ValueError(f"{COUNTRY} not found in border data")
    country.to_file(BORDER_CACHE, driver="GeoJSON")
    log.info("Border cached to %s", BORDER_CACHE)
    return country


def frange(start, stop, step):
    values, current = [], start
    while current <= stop + 1e-9:
        values.append(round(current, 4))
        current += step
    return values


def point_id(lat, lon):
    return f"GRID_{lat:.1f}_{lon:.1f}".replace(".", "p")


def run():
    border = load_border()
    shape = border.geometry.union_all()

    lats = frange(LAT_MIN, LAT_MAX, STEP)
    lons = frange(LON_MIN, LON_MAX, STEP)
    log.info("Grid frame: %d x %d = %d candidates", len(lats), len(lons), len(lats) * len(lons))

    inside, outside = [], []

    for lat in lats:
        for lon in lons:
            record = {"id": point_id(lat, lon), "latitude": lat, "longitude": lon}
            if shape.contains(Point(lon, lat)):
                inside.append(record)
            else:
                outside.append(record)

    report = {
        "source": "Natural Earth ne_10m_admin_0_countries",
        "country": COUNTRY,
        "bounds": {"lat": [LAT_MIN, LAT_MAX], "lon": [LON_MIN, LON_MAX], "step": STEP},
        "n_candidates": len(inside) + len(outside),
        "n_inside": len(inside),
        "n_outside": len(outside),
        "grid_points": inside,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    log.info("Done. candidates=%d inside=%d outside=%d",
             report["n_candidates"], len(inside), len(outside))
    log.info("Written to %s", OUT_PATH)

    return report


if __name__ == "__main__":
    run()
