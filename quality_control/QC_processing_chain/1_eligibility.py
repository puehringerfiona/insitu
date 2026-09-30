"""
Eligibility QC for hyperspectral in situ metadata tables.

This stage marks observations eligible only when they satisfy:
  1. spatial continuity: transect_nr is one of the manually selected transects;
  2. metadata completeness: timestamp, latitude, longitude, relative azimuth,
     and inclination metadata are present and valid.

Successful Rrs retrieval is intentionally not checked here. That criterion is
handled by later QC stages.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

INSITU_ROOT = Path(r"C:\Users\puehrifi\Documents\insitu")

METADATA_PATTERN = re.compile(
    r"^(?P<campaign>\d{8}_[A-Za-z0-9]+)_metadata_with_Rrs_(?P<sensor>L8|L9|S2A|S2B)(?:_[A-Za-z0-9]+)*\.csv$",
    re.IGNORECASE,
)

METADATA_GROUPS = {
    "timestamp": (
        "timestamp (UTC)",
        "timestamp",
        "datetime (UTC)",
        "datetime",
        "date_time",
        "utc_datetime",
        "measurement_time",
        "acquisition_time",
        "time",
        "date",
    ),
    "latitude": (
        "latitude",
        "lat",
        "gps_lat",
        "y-coordinate",
        "y_coordinate",
        "y",
    ),
    "longitude": (
        "longitude",
        "lon",
        "long",
        "gps_lon",
        "x-coordinate",
        "x_coordinate",
        "x",
    ),
    "relative_azimuth": (
        "relative azimuth angle",
        "relative_azimuth_angle",
        "relative_azimuth",
        "rel_azimuth",
        "vaz",
    ),
    "inclination": (
        "Incl_V",
        "incl_v",
        "inclination",
        "tilt",
        "pitch",
        "roll",
    ),
}


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


def normalize_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def to_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(series, errors="coerce")


def append_reason(current: pd.Series, mask: pd.Series | np.ndarray, reason: str) -> pd.Series:
    out = current.copy()
    mask = pd.Series(mask, index=out.index).fillna(False)
    out.loc[mask] = out.loc[mask].map(lambda old: f"{old};{reason}" if old else reason)
    return out


def combine_reasons(*parts: pd.Series) -> pd.Series:
    """Join semicolon-tagged reason strings, skipping empty parts.

    Each part is independently built by append_reason, so "" means that check
    raised no reason. Concatenating unconditionally would leave a stray leading
    ";" wherever the first part is empty.
    """
    combined = parts[0].fillna("").astype(str)
    for part in parts[1:]:
        part = part.fillna("").astype(str)
        both = combined.ne("") & part.ne("")
        either = combined.where(combined.ne(""), part)
        combined = (combined + ";" + part).where(both, either)
    return combined


def resolve_metadata_columns(df: pd.DataFrame) -> dict[str, str]:
    normalized_to_original = {normalize_name(col): str(col) for col in df.columns}
    resolved: dict[str, str] = {}
    for group, aliases in METADATA_GROUPS.items():
        for alias in aliases:
            normalized = normalize_name(alias)
            if normalized in normalized_to_original:
                resolved[group] = normalized_to_original[normalized]
                break
    return resolved


def valid_timestamp(values: pd.Series) -> pd.Series:
    present = values.notna() & values.astype(str).str.strip().ne("")
    numeric = to_numeric(values)

    if numeric.notna().mean() >= 0.8:
        return present & np.isfinite(numeric)

    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return present & parsed.notna()


def valid_numeric(values: pd.Series, *, lower: float | None = None, upper: float | None = None) -> pd.Series:
    present = values.notna() & values.astype(str).str.strip().ne("")
    numeric = to_numeric(values)
    valid = present & np.isfinite(numeric)
    if lower is not None:
        valid &= numeric >= lower
    if upper is not None:
        valid &= numeric <= upper
    return valid


def metadata_validity(df: pd.DataFrame, metadata_columns: dict[str, str]) -> tuple[pd.Series, pd.Series]:
    """(all_valid, reasons) - one reason tag per failing check, e.g. invalid_relative_azimuth."""
    validators = {
        "timestamp": lambda values: valid_timestamp(values),
        "latitude": lambda values: valid_numeric(values, lower=-90.0, upper=90.0),
        "longitude": lambda values: valid_numeric(values, lower=-180.0, upper=180.0),
        "relative_azimuth": lambda values: valid_numeric(values, lower=0.0, upper=360.0),
        "inclination": lambda values: valid_numeric(values),
    }

    reasons = pd.Series("", index=df.index, dtype="object")
    all_valid = pd.Series(True, index=df.index)

    for group, validator in validators.items():
        col = metadata_columns.get(group)
        missing_col = col is None
        if missing_col:
            valid = pd.Series(False, index=df.index)
        else:
            valid = validator(df[col])

        all_valid &= valid
        reason = f"missing_{group}_column" if missing_col else f"invalid_{group}"
        reasons = append_reason(reasons, ~valid, reason)

    return all_valid, reasons


def transect_validity(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(valid, reasons) for the spatial-continuity check."""
    reasons = pd.Series("", index=df.index, dtype="object")

    if "transect_nr" not in df.columns:
        valid = pd.Series(False, index=df.index)
        reasons = append_reason(reasons, ~valid, "missing_transect_nr_column")
    else:
        transect = to_numeric(df["transect_nr"])
        valid = np.isfinite(transect) & (transect >= 1)
        reasons = append_reason(reasons, ~valid, "invalid_transect_nr")

    return valid, reasons


def discover_hyperspectral_metadata_files(insitu_root: Path, campaign: str | None = None) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    if not insitu_root.exists():
        return files

    campaign_dirs = sorted(
        path for path in insitu_root.iterdir()
        if path.is_dir() and re.fullmatch(r"\d{8}_[A-Za-z0-9]+", path.name)
    )
    for campaign_dir in campaign_dirs:
        if campaign and campaign_dir.name != campaign:
            continue

        final_metadata = campaign_dir / "final_metadata"
        if not final_metadata.exists():
            continue

        for path in sorted(final_metadata.glob(f"{campaign_dir.name}_metadata_with_Rrs_*.csv")):
            match = METADATA_PATTERN.match(path.name)
            if match is None:
                continue
            files.append(
                {
                    "campaign": match.group("campaign"),
                    "sensor": match.group("sensor").upper(),
                    "path": path,
                }
            )

    return files


def output_path_for(input_csv: Path, campaign: str) -> Path:
    qc_dir = input_csv.parents[1] / "visualization_analysis" / "quality_control"
    return qc_dir / f"{input_csv.stem}_eligibility_qc.csv"


def process_file(input_csv: Path, campaign: str, delimiter: str | None = None) -> dict[str, object]:
    df, sep = read_table(input_csv, delimiter)

    metadata_columns = resolve_metadata_columns(df)
    spatial_valid, spatial_reasons = transect_validity(df)
    metadata_valid, metadata_reasons = metadata_validity(df, metadata_columns)

    result = df.copy()
    result["eligibility_reason"] = combine_reasons(spatial_reasons, metadata_reasons)
    result["eligibility_exclude"] = result["eligibility_reason"].ne("")

    output_csv = output_path_for(input_csv, campaign)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, sep=sep, index=False)

    return {
        "campaign": campaign,
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "rows": int(len(result)),
        "retained": int((~result["eligibility_exclude"]).sum()),
        "excluded": int(result["eligibility_exclude"].sum()),
        "spatial_continuity_fail": int((~spatial_valid).sum()),
        "metadata_completeness_fail": int((~metadata_valid).sum()),
        "metadata_columns_checked": ";".join(f"{key}:{value}" for key, value in metadata_columns.items()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply eligibility QC to hyperspectral metadata_with_Rrs tables.")
    parser.add_argument("--insitu-root", type=Path, default=INSITU_ROOT)
    parser.add_argument("--campaign", default=None, help="Optional campaign filter, e.g. 20230610_CST.")
    parser.add_argument("--input-csv", type=Path, default=None, help="Optional single metadata CSV to process.")
    parser.add_argument("--delimiter", default=None, help="Optional delimiter. Auto-detected by default.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input_csv is not None:
        match = METADATA_PATTERN.match(args.input_csv.name)
        if match is None:
            raise SystemExit(f"Input is not a canonical hyperspectral metadata file: {args.input_csv}")
        summary = process_file(args.input_csv, match.group("campaign"), delimiter=args.delimiter)
        print(
            f"Processed {summary['input_csv']}: "
            f"{summary['retained']} retained / {summary['rows']} rows -> {summary['output_csv']}"
        )
        return

    metadata_files = discover_hyperspectral_metadata_files(args.insitu_root, args.campaign)
    if not metadata_files:
        suffix = f" for campaign {args.campaign}" if args.campaign else ""
        raise SystemExit(f"No canonical hyperspectral metadata files found below {args.insitu_root}{suffix}")

    summaries: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for meta in metadata_files:
        try:
            summary = process_file(Path(meta["path"]), str(meta["campaign"]), delimiter=args.delimiter)
            summaries.append(summary)
            print(
                f"Processed {summary['input_csv']}: "
                f"{summary['retained']} retained / {summary['rows']} rows"
            )
        except Exception as exc:
            errors.append({"input_csv": str(meta["path"]), "error": str(exc)})
            print(f"Failed {meta['path']}: {exc}")

    print("Done.")
    print(f"Discovered metadata files: {len(metadata_files)}")
    print(f"Successfully processed: {len(summaries)}")
    print(f"Failed: {len(errors)}")
    if errors:
        for error in errors:
            print(f"{error['input_csv']}: {error['error']}")


if __name__ == "__main__":
    main()
