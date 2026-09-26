# Limitations: draft (pull into README at step 10)

## Design choices
- **Realistic replies from the harmful tools.** `send_report` returns "Report sent to X." and `delete_region_data` returns "Region data for X deleted.", even though both only write to an in-memory log. This is deliberate: a reply that said "simulated" would tell the agent it is in a sandbox, which could change its behaviour and bias the attack-success measurements.
- **`get_region_data` omits provenance.** The tool returns the measured values and the `notes` field, but not the scene lists and request URLs, to keep tool outputs short for the LLM's token budget. Full provenance stays in `data/regions.json`.
- **Keyword search.** `search_reports` is a case-insensitive keyword count with English stopwords removed, returning the top 2 reports. It is not semantic retrieval, so whether an injected report is retrieved depends on the words in the task prompt (see the `exposed` flag in the results).

## Radar water threshold (Sentinel-1 VV < −18 dB)
Water extent is a single global threshold on VV backscatter (a rough, commonly used heuristic). It is not calibrated per region. Smooth, dry bare soil also returns low VV backscatter and can be misclassified as water in arid regions. Example: in the test run, RJ-JAIPUR-01 showed a higher "water" fraction in the dry month (Feb 2025: 5.7%) than in the monsoon month (Aug 2025: 3.7%), which is the opposite of what real surface water would do.

## Optical data gaps during the monsoon
Sentinel-2 is optical, so monsoon cloud cover leaves months without a usable scene. A month is stored as `null` when the best fully covering scene (scene-level cloud cover < 80%) has < 50% cloud-free pixels over the AOI. These gaps cluster in Jun–Sep, exactly when vegetation and flooding change most. Per-region null counts are in `data/DATA_SOURCES.md`. Benign tasks only ask about non-null months.

## Flooded paddy
In paddy regions, fields are deliberately flooded. Both the Sentinel-1 radar threshold and Sentinel-2 SCL class 6 count this standing water as "water", so a high water % there reflects irrigation practice, not necessarily flooding. The same inundation lowers mean NDVI early in the season, independent of crop health.
- **Timing:** in PB-LUDHIANA-01, kharif paddy is transplanted from about mid-June and keeps standing water through about Jul–Aug. KL-KUTTANAD-01 lies below sea level and has surrounding water (canals, backwaters, flooded polders) for much of the year, not only at transplanting.
- **Consequence:** for PB-LUDHIANA-01 and KL-KUTTANAD-01, the monsoon water % mixes irrigation water and flood water. It is an **upper bound on flooding**, not a clean flood measure.
