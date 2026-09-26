"""Write 2 short analysis reports per region from data/regions.json (plain templates, no LLM).

  <ID>_vegetation.txt : Jun–Oct 2025 vegetation summary from Sentinel-2 NDVI (no crop-season claims)
  <ID>_monsoon.txt    : monsoon rainfall + water summary (Open-Meteo, Sentinel-1, Sentinel-2 SCL)

Every number in a report comes from regions.json (or is a sum/difference of its values). The
script verifies this: any number in the report body that was not produced by `num()` fails the build.

Run:  python scripts/build_reports.py
"""

from __future__ import annotations

import calendar
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "reports"
JUN_OCT = [f"2025-{m:02d}" for m in (6, 7, 8, 9, 10)]


class Numbers:
    """Formats numbers and remembers every string it produced, for the provenance check."""
    def __init__(self):
        self.allowed: set[str] = set()

    def __call__(self, value: float, dp: int, signed: bool = False) -> str:
        s = f"{value:+.{dp}f}" if signed else f"{value:.{dp}f}"
        self.allowed.add(s.lstrip("+-"))
        return s


def month_name(m: str) -> str:
    return f"{calendar.month_name[int(m[5:])]} {m[:4]}"


def s2_scene(r, m):
    return r["provenance"]["sentinel2"]["scenes"][m]


def vegetation_report(r, num: Numbers) -> tuple[str, list[str]]:
    ndvi = r["ndvi_monthly"]
    avail = {m: v for m, v in ndvi.items() if v is not None}
    peak = max(avail, key=avail.get)
    low = min(avail, key=avail.get)
    jo = [m for m in JUN_OCT if ndvi[m] is not None]
    jo_null = [m for m in JUN_OCT if ndvi[m] is None]

    lines = [
        f"{r['id']}: Jun–Oct vegetation summary, 2025",
        f"Region: {r['name']}, {r['state']}.",
        "",
        f"Across 2025, mean NDVI was highest in {month_name(peak)} ({num(avail[peak], 4)}) "
        f"and lowest in {month_name(low)} ({num(avail[low], 4)}).",
        "Monthly mean NDVI, June-October 2025:",
    ]
    for m in JUN_OCT:
        v = ndvi[m]
        lines.append(f"  - {month_name(m)}: " + (num(v, 4) if v is not None
                     else "no data (no usable cloud-free Sentinel-2 scene)"))
    if len(jo) >= 2:
        first, last = jo[0], jo[-1]
        lines.append(f"From {month_name(first)} to {month_name(last)}, mean NDVI changed by "
                     f"{num(ndvi[last] - ndvi[first], 4, signed=True)} "
                     f"({num(ndvi[first], 4)} to {num(ndvi[last], 4)}).")
    if jo_null:
        lines.append(f"Cloud cover during the monsoon left {len(jo_null)} of these months without "
                     f"usable optical data.")
    months_cited = sorted(set(jo) | {peak, low})
    sources = [s2_scene(r, m)["scene_id"] for m in months_cited]
    lines += ["", "Method: monthly mean NDVI = (B08 - B04) / (B08 + B04) over cloud-free pixels "
                  "(Sentinel-2 L2A scene classification mask), 20 m resolution."]
    return "\n".join(lines), ["Sentinel-2 L2A scenes " + ", ".join(sources)]


def monsoon_report(r, num: Numbers) -> tuple[str, list[str]]:
    rain = r["rain_mm_monthly"]
    season = r["monsoon_season_months"]
    season_label = f"{calendar.month_name[int(season[0][5:])]}-{calendar.month_name[int(season[-1][5:])]} 2025"
    season_total = sum(rain[m] for m in season)
    annual_total = sum(v for v in rain.values() if v is not None)
    wettest = max((m for m in rain if rain[m] is not None), key=lambda m: rain[m])
    s1 = r["provenance"]["sentinel1"]
    months = r["water_extent_months"]

    lines = [
        f"{r['id']}: Monsoon rainfall and water summary, 2025",
        f"Region: {r['name']}, {r['state']}.",
        "",
        f"Monsoon season for this region: {season_label}.",
        "Monthly rainfall during the monsoon season:",
    ]
    for m in season:
        lines.append(f"  - {month_name(m)}: {num(rain[m], 1)} mm")
    lines += [
        f"Monsoon-season total: {num(season_total, 1)} mm, out of {num(annual_total, 1)} mm for all of 2025.",
        f"The wettest month of 2025 was {month_name(wettest)} ({num(rain[wettest], 1)} mm).",
        "",
    ]
    dry, mon = r.get("water_extent_pct_dry"), r.get("water_extent_pct_monsoon")
    if dry is not None and mon is not None:
        lines.append(
            f"Sentinel-1 radar (VV < -18 dB) detected open water over {num(dry, 2)}% of the area in "
            f"{month_name(months['dry'])} (dry season) and {num(mon, 2)}% in {month_name(months['monsoon'])} "
            f"(monsoon), a change of {num(mon - dry, 2, signed=True)} percentage points.")
    else:
        lines.append("Sentinel-1 radar water extent is not available for both the dry and monsoon month.")
    if r.get("water_pct_scl_dry_season") is not None:
        lines.append(f"Sentinel-2 scene classification flagged {num(r['water_pct_scl_dry_season'], 2)}% "
                     f"of cloud-free pixels as water in January-March 2025 (mean).")
    lines += ["", "Note: the radar threshold detects open water only; water under crops or "
                  "vegetation is not counted."]

    s1_ids = [sid for lbl in ("dry", "monsoon") for s in s1[lbl].get("scenes", []) for sid in s["scene_ids"]]
    s2_ids = [s2_scene(r, m)["scene_id"] for m in ("2025-01", "2025-02", "2025-03")
              if r["water_pct_scl_monthly"][m] is not None]
    return "\n".join(lines), [
        "Open-Meteo Historical Weather API (daily precipitation_sum, " + r["provenance"]["rainfall"]["request_url"] + ")",
        "Sentinel-1 RTC scenes " + ", ".join(s1_ids),
        "Sentinel-2 L2A scenes " + ", ".join(s2_ids),
    ]


NUM_RE = re.compile(r"(?<![\w.-])\d+\.\d+(?![\w.])")   # decimals only: skips years, dates, IDs


def check_numbers(name: str, body: str, num: Numbers):
    bad = [x for x in NUM_RE.findall(body) if x not in num.allowed]
    if bad:
        raise SystemExit(f"{name}: numbers not traceable to regions.json: {bad}")


def main():
    records = json.loads((DATA / "regions.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.txt"):
        old.unlink()
    written = []
    for r in records:
        for kind, fn in (("vegetation", vegetation_report), ("monsoon", monsoon_report)):
            num = Numbers()
            body, sources = fn(r, num)
            check_numbers(f"{r['id']}_{kind}", body, num)
            text = body + "\n\nSources: " + "; ".join(sources) + "\n"
            path = OUT / f"{r['id']}_{kind}.txt"
            path.write_text(text, encoding="utf-8")
            written.append(path.name)
    print(f"wrote {len(written)} reports to {OUT}:")
    for n in written:
        print("  ", n)


if __name__ == "__main__":
    main()
