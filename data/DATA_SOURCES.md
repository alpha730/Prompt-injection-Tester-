# Data sources

All numbers in `data/regions.json` and `data/reports/` are real measurements for calendar year 2025, fetched by `scripts/fetch_real_data.py`. The reports are written from those numbers by `scripts/build_reports.py` using fixed templates (no LLM). No personal data is used. The data was downloaded on 2026-09-26, and `regions.json` was assembled at 2026-09-26 18:41 UTC.

## Regions (`data/aoi.geojson`)

Each area of interest (AOI) is a 5 × 5 km box.

| ID | Name | State | Centre (lat, lon) | BBox (W, S, E, N) |
|---|---|---|---|---|
| RJ-JAIPUR-01 | Chaksu farmland, south of Jaipur | Rajasthan | 26.62, 75.93 | 75.90488, 26.59754, 75.95512, 26.64246 |
| KL-KUTTANAD-01 | Kuttanad paddy polders (Champakulam area) | Kerala | 9.43, 76.42 | 76.39723, 9.40754, 76.44277, 9.45246 |
| AS-MAJULI-01 | Majuli island (Garamur area) | Assam | 26.95, 94.18 | 94.15481, 26.92754, 94.20519, 26.97246 |
| TN-CHENNAI-01 | Mudichur / Tambaram outskirts | Tamil Nadu | 12.92, 80.09 | 80.06696, 12.89754, 80.11304, 12.94246 |
| PB-LUDHIANA-01 | Farmland south-west of Ludhiana | Punjab | 30.80, 75.72 | 75.69385, 30.77754, 75.74615, 30.82246 |
| OD-PURI-01 | Inland coastal plain north of Puri | Odisha | 19.95, 85.80 | 85.77611, 19.92754, 85.82389, 19.97246 |

## 1. Sentinel-2 L2A (optical): monthly NDVI and SCL water %

- **Access:** Microsoft Planetary Computer STAC API (`https://planetarycomputer.microsoft.com/api/stac/v1`), collection `sentinel-2-l2a`. No account needed.
- **Scene selection, per region and month:** candidates are the scenes whose footprint fully contains the AOI and whose scene-level `eo:cloud_cover` is below 80%. The candidate with the highest AOI-level valid-pixel % is chosen. Ties go to the lowest `eo:cloud_cover`, then the earliest date. The month is kept only if at least 50% of the AOI is valid; otherwise it is stored as `null` with the reason.
- **Mask:** valid pixels are those whose scene classification (SCL) is not 0 (no data), 1 (saturated/defective), 3 (cloud shadow), 8 or 9 (cloud, medium/high probability), or 10 (thin cirrus).
- **NDVI:** mean of (B08 − B04) / (B08 + B04) over valid pixels, with bands loaded at 20 m. For scenes with processing baseline ≥ 04.00, the −1000 reflectance offset is removed first.
- **SCL water %:** the share of valid pixels in SCL class 6 (water). The dry-season check uses the January–March mean and warns above 20%. No region exceeds it; the highest is KL-KUTTANAD-01 at 9.60%.
- **Provenance per month** (scene ID, date, cloud cover, valid %, processing baseline) is stored in `regions.json` under `provenance.sentinel2`.

### Optical data gaps (null NDVI months)

6 of 72 region-months are null, all during the monsoon.

| Region | Null months | Months and reasons |
|---|---|---|
| RJ-JAIPUR-01 | 1 | 2025-07: best of 2 scenes has only 39.9% valid AOI pixels (< 50%) |
| KL-KUTTANAD-01 | 2 | 2025-06: best of 3 scenes has only 16.3% valid AOI pixels (< 50%); 2025-07: best of 1 scene has only 6.4% valid AOI pixels (< 50%) |
| AS-MAJULI-01 | 1 | 2025-08: best of 2 scenes has only 20.4% valid AOI pixels (< 50%) |
| TN-CHENNAI-01 | 0 | none |
| PB-LUDHIANA-01 | 0 | none |
| OD-PURI-01 | 2 | 2025-06: best of 3 scenes has only 37.1% valid AOI pixels (< 50%); 2025-07: no scene with eo:cloud_cover < 80 fully covering AOI |

The benign tasks only ask about non-null months.

**License and attribution:** Copernicus Sentinel data terms and conditions (free, full and open access). *Contains modified Copernicus Sentinel data 2025*, accessed via Microsoft Planetary Computer.

## 2. Sentinel-1 RTC (radar): open-water extent

- **Access:** Microsoft Planetary Computer STAC API, collection `sentinel-1-rtc` (radiometrically terrain-corrected backscatter; RTC processing by Catalyst). No account needed.
- **Method:** % of valid AOI pixels with VV backscatter < −18 dB, loaded at 20 m. This single threshold is a rough, commonly used open-water heuristic. It misses water under vegetation and can misclassify smooth dry soil as water (see the README limitations).
- **Passes:** adjacent frames from the same pass (same date, platform, orbit direction and relative orbit) are joined. A pass is used only if its joined frame footprints cover at least 99.5% of the AOI. The small tolerance allows for footprint outlines being simplified polygons; passes with less coverage are not used. The monthly value is the mean over all such passes. Scene IDs, orbit direction, relative orbit and footprint coverage % for each pass are stored under `provenance.sentinel1`.
- **Months (chosen in advance):**
  - The dry month is February 2025 for every region. If no pass meets the coverage rule, it falls back to March, then January 2025. February was available for all 6 regions.
  - The monsoon month is August 2025 for the five south-west-monsoon regions and November 2025 for TN-CHENNAI-01 (north-east monsoon).

| Region | Dry month used (months tried) | Dry water % (passes) | Monsoon month | Monsoon water % (passes) |
|---|---|---|---|---|
| RJ-JAIPUR-01 | 2025-02 (2025-02) | 5.69 (4) | 2025-08 | 3.66 (6) |
| KL-KUTTANAD-01 | 2025-02 (2025-02) | 2.08 (2) | 2025-08 | 13.38 (2) |
| AS-MAJULI-01 | 2025-02 (2025-02) | 0.14 (4) | 2025-08 | 0.36 (5) |
| TN-CHENNAI-01 | 2025-02 (2025-02) | 1.23 (3) | 2025-11 | 1.49 (3) |
| PB-LUDHIANA-01 | 2025-02 (2025-02) | 0.04 (8) | 2025-08 | 0.02 (11) |
| OD-PURI-01 | 2025-02 (2025-02) | 0.13 (2) | 2025-08 | 0.79 (3) |

The TN-CHENNAI-01 AOI lies on the boundary between two adjacent frames of the same descending pass (about 97% + 3%). Only after joining frames per pass does any pass cover it. Three of its passes are included only because of the 99.5% tolerance: 4 February (99.911%), 28 February (99.999%) and 7 November 2025 (99.921%). No other region's values depend on the tolerance.

**License and attribution:** CC BY 4.0 (collection license on Microsoft Planetary Computer; RTC processing by Catalyst). *Contains modified Copernicus Sentinel data 2025*, processed by Catalyst and accessed via Microsoft Planetary Computer.

## 3. Open-Meteo Historical Weather API: monthly rainfall

- **Endpoint:** `https://archive-api.open-meteo.com/v1/archive`, queried at the AOI centre with `daily=precipitation_sum`, `start_date=2025-01-01`, `end_date=2025-12-31`, `timezone=Asia/Kolkata`. No key needed.
- **Model:** Open-Meteo's default archive model (`best_match`, based on ERA5 / ERA5-Land reanalysis from the Copernicus Climate Change Service, C3S).
- **Aggregation:** daily totals summed to monthly totals (mm). No days were missing. The exact request URL for each region is stored under `provenance.rainfall`.
- **Monsoon season used for the sanity check and the reports (chosen in advance, matching the radar monsoon month):** June–September for the five south-west-monsoon regions, and October–December for TN-CHENNAI-01 (north-east monsoon). Stored per region as `monsoon_season_months`. With these definitions, the monsoon season is wetter than the rest of the year in all 6 regions.

**License and attribution:** CC BY 4.0. *Weather data by [Open-Meteo.com](https://open-meteo.com/)*. Underlying reanalysis: Copernicus Climate Change Service (C3S), ERA5.

## Sanity checks (output of `fetch_real_data.py`)

- All NDVI values lie in [−1, 1]. Every region shows seasonal variation; the annual NDVI range is 0.142 (TN-CHENNAI-01) to 0.492 (PB-LUDHIANA-01).
- The monsoon season (as defined above) is wetter than the rest of the year in 6 of 6 regions.
- No region has more than 4 null NDVI months (maximum 2).
- The SCL dry-season (Jan–Mar) water % is below 20% in every region.
