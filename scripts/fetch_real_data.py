"""Fetch real 2025 statistics for each AOI in data/aoi.geojson and write data/regions.json.

Sources:
  1. Sentinel-2 L2A (Microsoft Planetary Computer): monthly mean NDVI, SCL cloud mask,
     SCL water-pixel % (class 6).
  2. Sentinel-1 RTC (Microsoft Planetary Computer): % of AOI with VV < -18 dB, one dry
     and one monsoon month per region.
  3. Open-Meteo Historical Weather API: monthly rainfall totals.

Run once:  python scripts/fetch_real_data.py
Per-item results are cached in data/cache/ so an interrupted run can be resumed.
"""

from __future__ import annotations

import argparse
import calendar
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import planetary_computer
import pystac_client
import requests
from odc.stac import load as stac_load
from shapely.geometry import box, shape
from shapely.ops import unary_union

# PC applies the STAC `query` extension but doesn't advertise it; we also filter client-side below.
warnings.filterwarnings("ignore", message=".*does not conform to QUERY.*")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = DATA / "cache"

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
YEAR = 2025
MONTHS = [f"{YEAR}-{m:02d}" for m in range(1, 13)]

# Sentinel-2
S2_MAX_CLOUD = 80          # scene-level eo:cloud_cover prefilter
S2_MIN_VALID_PCT = 50.0    # keep a month only if >= 50% of the AOI is valid (cloud-free)
S2_RESOLUTION_M = 20
SCL_NODATA = 0
# saturated/defective, cloud shadow, cloud medium/high prob, thin cirrus
SCL_MASKED = {1, 3, 8, 9, 10}
SCL_WATER = 6
DRY_MONTHS = [f"{YEAR}-01", f"{YEAR}-02", f"{YEAR}-03"]  # for the SCL water check
DRY_WATER_WARN_PCT = 20.0

# Sentinel-1
S1_RESOLUTION_M = 20
S1_WATER_DB = -18.0
# Dry month: Feb; if no RTC pass fully covers the AOI, fall back to Mar, then Jan (all regions).
S1_DRY_MONTHS = [f"{YEAR}-02", f"{YEAR}-03", f"{YEAR}-01"]
S1_MONSOON_MONTH = {"default": f"{YEAR}-08", "TN-CHENNAI-01": f"{YEAR}-11"}  # NE monsoon for Chennai

# Each region's own monsoon season (chosen a priori): SW monsoon Jun-Sep; NE monsoon Oct-Dec for Chennai.
MONSOON_SEASON = {
    "default": [f"{YEAR}-{m:02d}" for m in (6, 7, 8, 9)],
    "TN-CHENNAI-01": [f"{YEAR}-{m:02d}" for m in (10, 11, 12)],
}


def monsoon_season(rid):
    return MONSOON_SEASON.get(rid, MONSOON_SEASON["default"])


# ---------------------------------------------------------------- helpers

def month_range(month: str) -> str:
    y, m = map(int, month.split("-"))
    return f"{month}-01/{month}-{calendar.monthrange(y, m)[1]:02d}"


def cached(key: str, fn):
    """Return cached JSON result for key, or compute and cache it."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    result = fn()
    path.write_text(json.dumps(result, indent=2))
    return result


def retry(fn, tries=4, wait=5.0):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # network hiccups, throttling, expired SAS tokens
            if i == tries - 1:
                raise
            print(f"    retry {i + 1}/{tries - 1} after error: {e!r}")
            time.sleep(wait * 2**i)


def epsg_of(item) -> str:
    p = item.properties
    if "proj:epsg" in p and p["proj:epsg"]:
        return f"EPSG:{p['proj:epsg']}"
    return p["proj:code"]


def covering_items(items, aoi_geom):
    """Only items whose footprint fully contains the AOI (no partial-tile edges)."""
    return [it for it in items if shape(it.geometry).contains(aoi_geom)]


def open_catalog():
    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


# ---------------------------------------------------------------- Sentinel-2

def scl_valid_mask(scl):
    return (scl != SCL_NODATA) & ~np.isin(scl, list(SCL_MASKED))


def load_s2(item, bbox, bands):
    resampling = {b: ("nearest" if b == "SCL" else "average") for b in bands}
    return stac_load([item], bands=bands, bbox=bbox, crs=epsg_of(item),
                     resolution=S2_RESOLUTION_M, resampling=resampling).isel(time=0)


def s2_month(catalog, region, month):
    """Pick the scene with the highest AOI-level valid % among fully covering scenes
    with eo:cloud_cover < S2_MAX_CLOUD; keep it only if valid >= S2_MIN_VALID_PCT."""
    bbox = region["bbox"]
    aoi = box(*bbox)
    search = catalog.search(
        collections=["sentinel-2-l2a"], bbox=bbox, datetime=month_range(month),
        query={"eo:cloud_cover": {"lt": S2_MAX_CLOUD}},
    )
    items = [it for it in covering_items(list(search.items()), aoi)
             if it.properties.get("eo:cloud_cover", 100) < S2_MAX_CLOUD]
    if not items:
        return {"ndvi_mean": None, "n_candidates": 0,
                "reason": f"no scene with eo:cloud_cover<{S2_MAX_CLOUD} fully covering AOI"}

    # Rank candidates by AOI-level valid % from SCL only (cheap: one 20 m band).
    candidates = []
    for it in items:
        it = planetary_computer.sign(it)
        scl = load_s2(it, bbox, ["SCL"])["SCL"].values
        candidates.append((100.0 * scl_valid_mask(scl).sum() / scl.size, it))
    # tie-break: lowest scene-level cloud cover, then earliest acquisition
    candidates.sort(key=lambda c: (-c[0], c[1].properties["eo:cloud_cover"], c[1].properties["datetime"]))
    best_valid, item = candidates[0]

    ds = load_s2(item, bbox, ["B04", "B08", "SCL"])
    red = ds["B04"].values.astype("float64")
    nir = ds["B08"].values.astype("float64")
    scl = ds["SCL"].values

    # Processing baseline >= 04.00 (Jan 2022 onward) adds a +1000 DN offset to L2A reflectance.
    baseline = str(item.properties.get("s2:processing_baseline", "0"))
    offset = 1000.0 if float(baseline) >= 4.0 else 0.0
    red -= offset
    nir -= offset

    valid = scl_valid_mask(scl) & (red + nir > 0)
    n_valid = int(valid.sum())
    valid_pct = round(100.0 * n_valid / scl.size, 1)
    rec = {
        "scene_id": item.id,
        "acquired": item.properties["datetime"][:10],
        "cloud_cover": round(float(item.properties["eo:cloud_cover"]), 2),
        "processing_baseline": baseline,
        "reflectance_offset_applied": offset,
        "n_candidates": len(candidates),
        "valid_pixel_pct": valid_pct,
        "water_pct_scl": round(100.0 * int(((scl == SCL_WATER) & valid).sum()) / n_valid, 2) if n_valid else None,
    }
    if valid_pct < S2_MIN_VALID_PCT:
        rec.update(ndvi_mean=None, water_pct_scl=None,
                   reason=f"best of {len(candidates)} scenes has only {valid_pct}% valid AOI pixels "
                          f"(< {S2_MIN_VALID_PCT:.0f}%)")
        return rec
    ndvi = (nir[valid] - red[valid]) / (nir[valid] + red[valid])
    rec["ndvi_mean"] = round(float(np.clip(ndvi, -1, 1).mean()), 4)
    return rec


# ---------------------------------------------------------------- Sentinel-1

def s1_month(catalog, region, month):
    bbox = region["bbox"]
    aoi = box(*bbox)
    # Group adjacent frames of the same pass (same date, platform, orbit direction, relative orbit);
    # a pass is used only if the union of its frames fully covers the AOI.
    passes = {}
    for it in catalog.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=month_range(month)).items():
        p = it.properties
        key = (p["datetime"][:10], (p.get("platform") or "").lower(), p.get("sat:orbit_state"), p.get("sat:relative_orbit"))
        passes.setdefault(key, []).append(it)
    covering = [(k, its) for k, its in passes.items()
                if unary_union([shape(it.geometry) for it in its]).contains(aoi)]
    if not covering:
        return {"water_extent_pct": None, "month": month,
                "reason": "no Sentinel-1 RTC pass (frames of one pass joined) fully covering AOI"}
    scenes = []
    for (date, platform, orbit_state, rel_orbit), its in sorted(covering, key=lambda c: c[0]):
        its = [planetary_computer.sign(it) for it in its]
        vv = stac_load(its, bands=["vv"], bbox=bbox, crs=epsg_of(its[0]), groupby="solar_day",
                       resolution=S1_RESOLUTION_M, resampling="average").isel(time=0)["vv"].values
        vv = vv.astype("float64")
        ok = np.isfinite(vv) & (vv > 0)
        if not ok.any():
            continue
        db = 10.0 * np.log10(vv[ok])
        scenes.append({
            "scene_ids": sorted(it.id for it in its),   # >1 when adjacent frames were joined
            "acquired": date,
            "platform": platform,
            "orbit_direction": orbit_state,
            "relative_orbit": rel_orbit,
            "valid_pixel_pct": round(100.0 * ok.sum() / vv.size, 1),
            "water_extent_pct": round(100.0 * float((db < S1_WATER_DB).mean()), 2),
        })
    if not scenes:
        return {"water_extent_pct": None, "month": month, "reason": "all covering scenes empty over AOI"}
    return {
        "month": month,
        # headline value = mean over every fully covering acquisition in the month
        "water_extent_pct": round(float(np.mean([s["water_extent_pct"] for s in scenes])), 2),
        "scenes": scenes,
    }


# ---------------------------------------------------------------- Open-Meteo

def rainfall(region):
    lon, lat = region["centroid"]
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": f"{YEAR}-01-01", "end_date": f"{YEAR}-12-31",
        "daily": "precipitation_sum", "timezone": "Asia/Kolkata",
    }
    r = requests.get(OPEN_METEO_URL, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    monthly = {m: 0.0 for m in MONTHS}
    missing = {m: 0 for m in MONTHS}
    for day, val in zip(js["daily"]["time"], js["daily"]["precipitation_sum"]):
        if val is None:
            missing[day[:7]] += 1
        else:
            monthly[day[:7]] += val
    return {
        "rain_mm_monthly": {m: (round(v, 1) if missing[m] == 0 else None) for m, v in monthly.items()},
        "missing_days": {m: n for m, n in missing.items() if n},
        "request_url": r.url,
        "response_meta": {k: js.get(k) for k in ("latitude", "longitude", "elevation", "timezone", "daily_units")},
        "model": "Open-Meteo archive default (best_match; ERA5 / ERA5-Land reanalysis)",
    }


# ---------------------------------------------------------------- main

def load_regions(only):
    fc = json.loads((DATA / "aoi.geojson").read_text())
    regions = []
    for f in fc["features"]:
        p = f["properties"]
        if only and p["id"] not in only:
            continue
        regions.append({**p, "bbox": list(shape(f["geometry"]).bounds)})
    return regions


def fetch_region(catalog, region, skip_s1):
    rid = region["id"]
    print(f"\n== {rid}")
    s2 = {}
    for m in MONTHS:
        s2[m] = cached(f"s2_{rid}_{m}", lambda m=m: retry(lambda: s2_month(catalog, region, m)))
        r = s2[m]
        print(f"  S2 {m}: ndvi={r.get('ndvi_mean')} valid={r.get('valid_pixel_pct')}% "
              f"cc={r.get('cloud_cover')} water={r.get('water_pct_scl')}% {r.get('reason', '')}")

    s1 = None
    if not skip_s1:
        monsoon_month = S1_MONSOON_MONTH.get(rid, S1_MONSOON_MONTH["default"])
        s1 = {}
        dry_tried = []
        for m in S1_DRY_MONTHS:
            s1["dry"] = cached(f"s1_{rid}_{m}", lambda m=m: retry(lambda: s1_month(catalog, region, m)))
            dry_tried.append({"month": m, "result": "used" if s1["dry"].get("scenes") else s1["dry"].get("reason")})
            if s1["dry"].get("scenes"):
                break
        s1["dry"]["months_tried"] = dry_tried
        s1["monsoon"] = cached(f"s1_{rid}_{monsoon_month}",
                               lambda: retry(lambda: s1_month(catalog, region, monsoon_month)))
        for label in ("dry", "monsoon"):
            print(f"  S1 {label} ({s1[label]['month']}): water={s1[label].get('water_extent_pct')}% "
                  f"scenes={len(s1[label].get('scenes', []))} {s1[label].get('reason', '')}")

    rain = cached(f"rain_{rid}", lambda: retry(lambda: rainfall(region)))
    print(f"  rain: {rain['rain_mm_monthly']}")

    dry_water = [s2[m]["water_pct_scl"] for m in DRY_MONTHS if s2[m].get("water_pct_scl") is not None]

    rec = {
        "id": rid,
        "name": region["name"],
        "state": region["state"],
        "description": region.get("description"),
        "bbox": [round(x, 5) for x in region["bbox"]],
        "centroid": region["centroid"],
        "ndvi_monthly": {m: s2[m].get("ndvi_mean") for m in MONTHS},
        "cloud_cover_monthly": {m: s2[m].get("cloud_cover") for m in MONTHS},
        "valid_pixel_pct_monthly": {m: s2[m].get("valid_pixel_pct") for m in MONTHS},
        "water_pct_scl_monthly": {m: s2[m].get("water_pct_scl") for m in MONTHS},
        "water_pct_scl_dry_season": round(float(np.mean(dry_water)), 2) if dry_water else None,
        "rain_mm_monthly": rain["rain_mm_monthly"],
        "monsoon_season_months": monsoon_season(rid),
    }
    if s1 is not None:
        rec["water_extent_pct_dry"] = s1["dry"]["water_extent_pct"]
        rec["water_extent_pct_monsoon"] = s1["monsoon"]["water_extent_pct"]
        # month actually used (null if every fallback month failed)
        rec["water_extent_months"] = {lbl: (s1[lbl]["month"] if s1[lbl].get("scenes") else None)
                                      for lbl in ("dry", "monsoon")}
    rec["provenance"] = {
        "sentinel2": {
            "collection": "sentinel-2-l2a", "stac_api": STAC_URL,
            "selection": (f"among scenes fully covering the AOI with eo:cloud_cover < {S2_MAX_CLOUD}, "
                          f"the one with the highest AOI valid-pixel %; month kept only if valid >= "
                          f"{S2_MIN_VALID_PCT:.0f}% (ties: lowest eo:cloud_cover, then earliest)"),
            "resolution_m": S2_RESOLUTION_M,
            "masked_scl_classes": sorted(SCL_MASKED | {SCL_NODATA}),
            "scenes": {m: {k: v for k, v in s2[m].items() if k != "ndvi_mean"} for m in MONTHS},
        },
        "rainfall": {k: v for k, v in rain.items() if k != "rain_mm_monthly"},
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if s1 is not None:
        rec["provenance"]["sentinel1"] = {
            "collection": "sentinel-1-rtc", "stac_api": STAC_URL, "threshold_db": S1_WATER_DB,
            "polarisation": "VV", "resolution_m": S1_RESOLUTION_M,
            "value": "mean over all passes in the month whose frames (joined per pass) fully cover the AOI", **s1,
        }
    rec["notes"] = ""
    return rec


def sanity_checks(records):
    print("\n================ SANITY CHECKS ================")
    problems = 0
    for r in records:
        rid = r["id"]
        nd = {m: v for m, v in r["ndvi_monthly"].items() if v is not None}
        nulls = [m for m, v in r["ndvi_monthly"].items() if v is None]
        out_of_range = [m for m, v in nd.items() if not -1 <= v <= 1]
        rng = (max(nd.values()) - min(nd.values())) if nd else float("nan")
        rain = r["rain_mm_monthly"]
        season = r["monsoon_season_months"]
        mons = [rain[m] for m in season if rain[m] is not None]
        rest = [rain[m] for m in MONTHS if m not in season and rain[m] is not None]
        mons_avg = np.mean(mons) if mons else float("nan")
        rest_avg = np.mean(rest) if rest else float("nan")

        print(f"\n{rid}")
        print(f"  NDVI: {len(nd)}/12 months, min={min(nd.values()) if nd else None}, "
              f"max={max(nd.values()) if nd else None}, seasonal range={rng:.3f}")
        if out_of_range:
            print(f"  !! NDVI outside [-1,1] in {out_of_range}"); problems += 1
        if len(nulls) > 4:
            print(f"  !! {len(nulls)} null NDVI months ({nulls}) -> ask whether to swap this region"); problems += 1
        elif nulls:
            print(f"  null NDVI months: {nulls}")
        flag = "" if mons_avg > rest_avg else "  !! monsoon NOT wetter than rest of year"
        if flag:
            problems += 1
        label = f"{calendar.month_abbr[int(season[0][5:])]}-{calendar.month_abbr[int(season[-1][5:])]}"
        print(f"  Rain: monsoon ({label}) avg {mons_avg:.1f} mm/month vs other months {rest_avg:.1f}{flag}")
        dw = r["water_pct_scl_dry_season"]
        if dw is None:
            print("  !! SCL dry-season water %: no valid Jan-Mar scenes"); problems += 1
        else:
            warn = f"  !! above {DRY_WATER_WARN_PCT:.0f}% WARNING" if dw > DRY_WATER_WARN_PCT else ""
            print(f"  SCL water (class 6), Jan-Mar mean: {dw:.2f}%{warn}")
            problems += bool(warn)
        if "water_extent_pct_dry" in r:
            print(f"  S1 water extent: dry={r['water_extent_pct_dry']}% "
                  f"monsoon={r['water_extent_pct_monsoon']}% ({r['water_extent_months']})")
    def wetter(r):
        rain, season = r["rain_mm_monthly"], r["monsoon_season_months"]
        return (np.mean([rain[m] for m in season if rain[m] is not None])
                > np.mean([rain[m] for m in MONTHS if m not in season and rain[m] is not None]))
    n_wet = sum(wetter(r) for r in records)
    print(f"\nOwn monsoon season wetter than rest of year in {n_wet}/{len(records)} regions "
          f"(Jun-Sep; Oct-Dec for TN-CHENNAI-01)")
    print(f"Flags raised: {problems}")

    print("\n================ NULL NDVI MONTHS ================")
    print("| Region | # null | Null months |\n|---|---|---|")
    for r in records:
        nulls = [m[5:] for m, v in r["ndvi_monthly"].items() if v is None]
        print(f"| {r['id']} | {len(nulls)} | {', '.join(nulls) or '-'} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="*", help="only these region IDs (default: all)")
    ap.add_argument("--skip-s1", action="store_true", help="skip Sentinel-1 radar water extent")
    ap.add_argument("--out", default=str(DATA / "regions.json"))
    args = ap.parse_args()

    regions = load_regions(args.regions)
    if not regions:
        sys.exit("no regions matched")
    catalog = open_catalog()
    records = [fetch_region(catalog, r, args.skip_s1) for r in regions]
    Path(args.out).write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out}")
    sanity_checks(records)


if __name__ == "__main__":
    main()
