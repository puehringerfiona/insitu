"""
Diagnostic metrics and soft flags for hard-filtered in situ Rrs tables.

This stage adds acquisition geometry metrics, temporal matchup offsets, and
campaign-level quality flags. Metrics are retained as separate diagnostic
columns; suitability is represented by binary flags where the method defines a
suitable range.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd


INSITU_ROOT = Path(r"C:\Users\puehrifi\Documents\insitu")
TRANSECT_TIMEFRAMES_CSV = INSITU_ROOT / "all" / "transect_timeframes_cet.csv"

HARD_FILTER_PATTERN = re.compile(
    r"^(?P<campaign>\d{8}_[A-Za-z0-9]+)_metadata_with_Rrs_(?P<sensor>L8|L9|S2A|S2B)(?:_[A-Za-z0-9]+)*_hard_filter_qc\.csv$",
    re.IGNORECASE,
)

VZA_DEG = 40.0

VAZ_ACCEPT_LOW = 75.0
VAZ_ACCEPT_HIGH = 150.0
VAZ_STRICT_LOW = 90.0
VAZ_STRICT_HIGH = 135.0

SCATTERING_SUITABLE_LOW = 120.0
SCATTERING_SUITABLE_HIGH = 140.0

INCL_SUITABLE_MAX_DEG = 5.0

INCLINATION_BIAS_CAMPAIGNS = {"20260227_ZRH", "20260423_ZRH", "20260430_ZRH"}

SOFT_FLAG_COLUMNS = (
    "flag_vaz_outside_75_150",
    "flag_vaz_outside_90_135",
    "flag_scattering_angle_outside_120_140",
    "flag_inclination_gt_5deg",
    "flag_campaign_inclination_bias",
)

# Not written to the output CSV; each is either an exact duplicate produced
# elsewhere, unread everywhere, or trivially recomputable from a kept column.
OUTPUT_DROP_COLUMNS = [
    # Third copy of the "relative azimuth angle" raw column (also duplicated,
    # under a kept name, as hard_filter_vaz_deg). radiometry_transect_heatmaps.py's
    # fallback chain already prefers hard_filter_vaz_deg first.
    "metric_relative_azimuth_deg",
    # Bit-identical to hard_filter_vaz_outside_75_150 (kept) - same formula, same
    # 75/150 bounds, computed twice across two stages.
    "flag_vaz_outside_75_150",
    # No consumer reads this column; common.py hardcodes its own EXCLUDED_CAMPAIGNS
    # list instead.
    "flag_campaign_inclination_bias",
    # Always True; pure bookkeeping.
    "flags_evaluated",
    # Unread; trivially recomputable as a sum of the kept flag_* columns.
    "flag_count_without_hard_filter",
    "flag_count_incl_hard_filter",
]


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
        sample = handle.read(16384)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample else ""
        return ";" if first_line.count(";") > first_line.count(",") else ","


def read_table(path: Path, delimiter: str | None = None) -> tuple[pd.DataFrame, str]:
    sep = delimiter or detect_delimiter(path)
    df = pd.read_csv(path, sep=sep, low_memory=False)
    df.columns = [str(col).strip().replace("\ufeff", "") for col in df.columns]
    return df, sep


def to_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(series, errors="coerce")


def parse_metadata_times(df: pd.DataFrame) -> pd.Series:
    if "timestamp (UTC)" in df.columns:
        return pd.to_datetime(to_numeric(df["timestamp (UTC)"]), unit="s", errors="coerce", utc=True)
    if "datetime (UTC)" in df.columns:
        return pd.to_datetime(df["datetime (UTC)"], errors="coerce", utc=True, dayfirst=True)
    return pd.to_datetime(df["datetime (CET)"], errors="coerce", utc=True, dayfirst=True)


def local_naive_times(df: pd.DataFrame) -> pd.Series:
    if "datetime (CET)" in df.columns:
        return pd.to_datetime(df["datetime (CET)"], errors="coerce", utc=True, dayfirst=True).dt.tz_convert("Europe/Zurich").dt.tz_localize(None)
    return parse_metadata_times(df).dt.tz_convert("Europe/Zurich").dt.tz_localize(None)


def compute_scattering_angle(sza_deg, vza_deg: float, vaz_deg) -> np.ndarray:
    sza = np.deg2rad(np.asarray(sza_deg, dtype=float))
    vza = np.deg2rad(vza_deg)
    vaz = np.deg2rad(np.asarray(vaz_deg, dtype=float))
    cos_theta = -np.cos(sza) * np.cos(vza) + np.sin(sza) * np.sin(vza) * np.cos(vaz)
    return np.rad2deg(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def compute_geometry_indicators(df: pd.DataFrame) -> pd.DataFrame:
    qc = pd.DataFrame(index=df.index)
    vaz = to_numeric(df["relative azimuth angle"]) if "relative azimuth angle" in df.columns else pd.Series(np.nan, index=df.index)
    sza = to_numeric(df["instantaneous solar zenith angle"]) if "instantaneous solar zenith angle" in df.columns else pd.Series(np.nan, index=df.index)
    qc["metric_relative_azimuth_deg"] = vaz
    qc["flag_vaz_outside_75_150"] = ~(np.isfinite(vaz) & vaz.between(VAZ_ACCEPT_LOW, VAZ_ACCEPT_HIGH))
    qc["flag_vaz_outside_90_135"] = ~(np.isfinite(vaz) & vaz.between(VAZ_STRICT_LOW, VAZ_STRICT_HIGH))
    qc["metric_scattering_angle_deg"] = compute_scattering_angle(sza, VZA_DEG, vaz)
    qc["flag_scattering_angle_outside_120_140"] = ~(
        np.isfinite(qc["metric_scattering_angle_deg"])
        & qc["metric_scattering_angle_deg"].between(SCATTERING_SUITABLE_LOW, SCATTERING_SUITABLE_HIGH)
    )
    if "Incl_V" in df.columns:
        incl = to_numeric(df["Incl_V"]).abs()
    else:
        incl = pd.Series(np.nan, index=df.index)
    qc["metric_inclination_deg"] = incl
    qc["flag_inclination_gt_5deg"] = ~(np.isfinite(incl) & (incl <= INCL_SUITABLE_MAX_DEG))
    return qc


def compute_campaign_flags(index: pd.Index, campaign: str) -> pd.DataFrame:
    qc = pd.DataFrame(index=index)
    qc["flag_campaign_inclination_bias"] = campaign in INCLINATION_BIAS_CAMPAIGNS
    return qc


def compute_flags(df: pd.DataFrame, campaign_dir: Path, campaign: str, sensor: str, timeframes_csv: Path) -> pd.DataFrame:
    geometry = compute_geometry_indicators(df)
    temporal = pd.DataFrame(index=df.index)
    temporal["metric_temporal_offset_min"] = to_numeric(df["temporal_offset_min"]) if "temporal_offset_min" in df.columns else np.nan
    campaign_flags = compute_campaign_flags(df.index, campaign)
    result = pd.concat([df.copy(), geometry, temporal, campaign_flags], axis=1)
    result["flags_evaluated"] = True
    result["flag_count_without_hard_filter"] = result[list(SOFT_FLAG_COLUMNS)].astype(bool).sum(axis=1).astype(int)
    if "hard_filter_exclude" in result.columns:
        hard_filter_exclude = result["hard_filter_exclude"].astype("boolean").fillna(False).astype(bool)
    else:
        hard_filter_exclude = pd.Series(False, index=result.index)
    result["flag_count_incl_hard_filter"] = result["flag_count_without_hard_filter"] + hard_filter_exclude.astype(int)
    return result


def discover_hard_filter_files(insitu_root: Path, campaign: str | None = None) -> list[dict[str, object]]:
    files = []
    for campaign_dir in sorted(path for path in insitu_root.iterdir() if path.is_dir() and re.fullmatch(r"\d{8}_[A-Za-z0-9]+", path.name)):
        if campaign and campaign_dir.name != campaign:
            continue
        qc_dir = campaign_dir / "visualization_analysis" / "quality_control"
        for path in sorted(qc_dir.glob(f"{campaign_dir.name}_metadata_with_Rrs_*_hard_filter_qc.csv")) if qc_dir.exists() else []:
            match = HARD_FILTER_PATTERN.match(path.name)
            if match:
                files.append({"campaign": match.group("campaign"), "sensor": match.group("sensor").upper(), "path": path, "campaign_dir": campaign_dir})
    return files


def output_path_for(input_csv: Path) -> Path:
    return input_csv.with_name(input_csv.name.replace("_hard_filter_qc.csv", "_flags_qc.csv"))


def process_file(meta: dict[str, object], delimiter: str | None, timeframes_csv: Path) -> dict[str, object]:
    df, sep = read_table(Path(meta["path"]), delimiter)
    result = compute_flags(df, Path(meta["campaign_dir"]), str(meta["campaign"]), str(meta["sensor"]), timeframes_csv)
    # Summary counts below are computed from the full result, before trimming
    # columns that aren't written to the output CSV (see OUTPUT_DROP_COLUMNS).
    output_csv = output_path_for(Path(meta["path"]))
    output_df = result.drop(columns=OUTPUT_DROP_COLUMNS, errors="ignore")
    output_df.to_csv(output_csv, sep=sep, index=False)
    return {
        "campaign": meta["campaign"],
        "sensor": meta["sensor"],
        "input_csv": str(meta["path"]),
        "output_csv": str(output_csv),
        "rows": int(len(result)),
        "evaluated": int(result["flags_evaluated"].sum()),
        "vaz_outside_75_150": int(result["flag_vaz_outside_75_150"].sum()),
        "vaz_outside_90_135": int(result["flag_vaz_outside_90_135"].sum()),
        "scattering_angle_outside_120_140": int(result["flag_scattering_angle_outside_120_140"].sum()),
        "inclination_gt_5deg": int(result["flag_inclination_gt_5deg"].sum()),
        "campaign_inclination_bias": int(result["flag_campaign_inclination_bias"].sum()),
        "rows_with_any_flag_without_hard_filter": int((result["flag_count_without_hard_filter"] > 0).sum()),
        "rows_with_any_flag_incl_hard_filter": int((result["flag_count_incl_hard_filter"] > 0).sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute quality indicators and soft flags for hard-filtered Rrs metadata.")
    parser.add_argument("--insitu-root", type=Path, default=INSITU_ROOT)
    parser.add_argument("--transect-timeframes-csv", type=Path, default=TRANSECT_TIMEFRAMES_CSV)
    parser.add_argument("--campaign", default=None)
    parser.add_argument("--delimiter", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = discover_hard_filter_files(args.insitu_root, args.campaign)
    if not files:
        suffix = f" for campaign {args.campaign}" if args.campaign else ""
        raise SystemExit(f"No hard-filter QC files found below {args.insitu_root}{suffix}")
    summaries = []
    errors = []
    for meta in files:
        try:
            summary = process_file(meta, args.delimiter, args.transect_timeframes_csv)
            summaries.append(summary)
            print(f"Processed {summary['input_csv']}: {summary['evaluated']} evaluated / {summary['rows']} rows")
        except Exception as exc:
            errors.append({"input_csv": str(meta["path"]), "error": str(exc)})
            print(f"Failed {meta['path']}: {exc}")
    print("Done.")
    print(f"Discovered hard-filter files: {len(files)}")
    print(f"Successfully processed: {len(summaries)}")
    print(f"Failed: {len(errors)}")
    if errors:
        for error in errors:
            print(f"{error['input_csv']}: {error['error']}")


if __name__ == "__main__":
    main()
