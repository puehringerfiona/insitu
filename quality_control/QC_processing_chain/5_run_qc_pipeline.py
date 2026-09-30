"""
Run the hyperspectral in situ QC processing chain.

Pipeline:
  1. eligibility QC from final_metadata metadata_with_Rrs CSV files
  2. hard exclusion filters
  3. bottom-reflectance screening
  4. merge bottom-reflectance hard criterion into hard-filter QC
  5. quality flags
  6. final metadata QC table
  7. propagate final QC to convolved/current metadata files
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import shutil
import sys
from pathlib import Path

import pandas as pd


INSITU_ROOT = Path(r"C:\Users\puehrifi\Documents\insitu")
EO_INPUT_ROOT = Path(r"C:\Users\puehrifi\Documents\eo_data\input")
SCRIPT_DIR = Path(__file__).resolve().parent
TRANSECT_TIMEFRAMES_CSV = INSITU_ROOT / "all" / "transect_timeframes_cet.csv"

METADATA_PATTERN = re.compile(
    r"^(?P<campaign>\d{8}_[A-Za-z0-9]+)_metadata_with_Rrs_(?P<sensor>L8|L9|S2A|S2B)(?:_[A-Za-z0-9]+)*\.csv$",
    re.IGNORECASE,
)

BOTTOM_HARD_REASONS = {"bottom_reflectance_risk", "bottom_reflectance_unavailable"}


def load_stage_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


eligibility = load_stage_module("1_eligibility.py", "qc_stage_eligibility")
hard_filter = load_stage_module("2_hard_filter.py", "qc_stage_hard_filter")
bottom_reflectance = load_stage_module("3_bottom_reflectance.py", "qc_stage_bottom_reflectance")
flags = load_stage_module("4_flags.py", "qc_stage_flags")
propagate_qc = load_stage_module("6_propagate_qc_to_convolved.py", "qc_stage_propagate")


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


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "yes", "y"})


def discover_metadata_files(
    insitu_root: Path,
    campaign: str | None,
    input_csv: Path | None,
    current_only: bool = False,
) -> list[dict[str, object]]:
    if input_csv is not None:
        match = METADATA_PATTERN.match(input_csv.name)
        if match is None:
            raise ValueError(f"Input is not a canonical hyperspectral metadata file: {input_csv}")
        return [{"campaign": match.group("campaign"), "sensor": match.group("sensor").upper(), "path": input_csv}]

    files: list[dict[str, object]] = []
    for campaign_dir in sorted(path for path in insitu_root.iterdir() if path.is_dir() and re.fullmatch(r"\d{8}_[A-Za-z0-9]+", path.name)):
        if campaign and campaign_dir.name != campaign:
            continue
        final_metadata = campaign_dir / "final_metadata"
        if not final_metadata.exists():
            continue
        for path in sorted(final_metadata.glob(f"{campaign_dir.name}_metadata_with_Rrs_*.csv")):
            if any(token in path.name for token in ("_example", "_old")):
                continue
            if current_only:
                if "_current" not in path.name:
                    continue
            elif any(token in path.name for token in ("_conv_", "_current")):
                continue
            match = METADATA_PATTERN.match(path.name)
            if match:
                files.append({"campaign": match.group("campaign"), "sensor": match.group("sensor").upper(), "path": path})
    return files


def bottom_output_path_for(hard_filter_csv: Path) -> Path:
    base = hard_filter_csv.stem.replace("_hard_filter_qc", "")
    return hard_filter_csv.parent / "bottom_reflectance" / f"{base}_bottom_reflectance_qc.csv"


def final_output_path_for(input_csv: Path) -> Path:
    qc_dir = input_csv.parents[1] / "visualization_analysis" / "quality_control"
    return qc_dir / f"{input_csv.stem}_final_qc.csv"


def remove_reasons(reason_text: object, reasons_to_remove: set[str]) -> str:
    reasons = [part for part in str(reason_text).split(";") if part and part not in reasons_to_remove and part.lower() != "nan"]
    return ";".join(reasons)


def append_reason(current: pd.Series, mask: pd.Series, reason: str) -> pd.Series:
    out = current.fillna("").astype(str).map(lambda value: remove_reasons(value, set()))
    mask = mask.fillna(False).astype(bool)
    out.loc[mask] = out.loc[mask].map(lambda old: f"{old};{reason}" if old else reason)
    return out


def merge_bottom_into_hard_filter(hard_filter_csv: Path, bottom_csv: Path, delimiter: str | None = None) -> dict[str, object]:
    hard_df, sep = read_table(hard_filter_csv, delimiter)
    bottom_df, _ = read_table(bottom_csv, delimiter)

    # "bottom_" also matches the legacy "bottom_reflectance_*" columns, so a
    # re-run over an older hard-filter table drops them instead of duplicating.
    bottom_columns = [col for col in bottom_df.columns if col.startswith("bottom_")]
    if not bottom_columns:
        raise ValueError(f"No bottom_* columns found in {bottom_csv}")

    merge_key = None
    for candidate in ("fid_ramses", "timestamp (UTC)", "datetime (UTC)", "datetime (CET)"):
        if candidate in hard_df.columns and candidate in bottom_df.columns:
            merge_key = candidate
            break
    if merge_key is None:
        raise ValueError(f"No shared merge key found for {hard_filter_csv} and {bottom_csv}")

    bottom_subset = bottom_df[[merge_key] + bottom_columns].drop_duplicates(subset=[merge_key])
    stale = [col for col in hard_df.columns if col.startswith("bottom_")]
    hard_df = hard_df.drop(columns=stale, errors="ignore")
    merged = hard_df.merge(bottom_subset, on=merge_key, how="left")

    bottom_unavailable = merged["bottom_class"].isna() if "bottom_class" in merged.columns else pd.Series(True, index=merged.index)
    bottom_exclude = to_bool(merged["bottom_exclude"]) if "bottom_exclude" in merged.columns else pd.Series(False, index=merged.index)

    reasons = merged["hard_filter_exclusion_reason"].fillna("").astype(str).map(lambda value: remove_reasons(value, BOTTOM_HARD_REASONS))
    reasons = append_reason(reasons, bottom_exclude, "bottom_reflectance_risk")
    reasons = append_reason(reasons, bottom_unavailable, "bottom_reflectance_unavailable")
    merged["hard_filter_exclusion_reason"] = reasons
    merged["hard_filter_bottom_reflectance_exclude"] = bottom_exclude
    merged["hard_filter_bottom_reflectance_unavailable"] = bottom_unavailable
    merged["hard_filter_exclude"] = merged["hard_filter_exclusion_reason"].ne("")
    merged["hard_filter_retain"] = ~merged["hard_filter_exclude"]
    if "hard_filter_exclusion_reason_strict" in merged.columns:
        strict_reasons = merged["hard_filter_exclusion_reason_strict"].fillna("").astype(str).map(lambda value: remove_reasons(value, BOTTOM_HARD_REASONS))
        strict_reasons = append_reason(strict_reasons, bottom_exclude, "bottom_reflectance_risk")
        strict_reasons = append_reason(strict_reasons, bottom_unavailable, "bottom_reflectance_unavailable")
        merged["hard_filter_exclusion_reason_strict"] = strict_reasons
        # hard_filter_exclude_strict is an exact negation of hard_filter_retain_strict and
        # has no consumers, so it is deliberately not materialised here.
        merged["hard_filter_retain_strict"] = merged["hard_filter_exclusion_reason_strict"].eq("")
    merged.to_csv(hard_filter_csv, sep=sep, index=False)
    return {
        "hard_filter_csv": str(hard_filter_csv),
        "bottom_csv": str(bottom_csv),
        "rows": int(len(merged)),
        "bottom_reflectance_excluded": int(bottom_exclude.sum()),
        "bottom_reflectance_unavailable": int(bottom_unavailable.sum()),
        "retained_after_bottom": int(merged["hard_filter_retain"].sum()),
        "retained_strict_after_bottom": int(merged["hard_filter_retain_strict"].sum()) if "hard_filter_retain_strict" in merged.columns else None,
    }


def run_bottom_reflectance_for_files(
    hard_filter_files: list[dict[str, object]],
    delimiter: str | None,
    secchi_table: Path | None,
) -> list[dict[str, object]]:
    secchi_by_campaign, secchi_source = bottom_reflectance.resolve_secchi_by_campaign(secchi_table)

    summaries = []
    for meta in hard_filter_files:
        secchi_depth_m = secchi_by_campaign.get(str(meta["campaign"]))
        item = bottom_reflectance.compute_bottom_qc(meta, secchi_depth_m, delimiter)
        item.summary_base["secchi_source"] = secchi_source
        summaries.append(bottom_reflectance.write_outputs(item))

    campaign_filter = str(hard_filter_files[0]["campaign"]) if hard_filter_files and len({str(meta["campaign"]) for meta in hard_filter_files}) == 1 else None
    bottom_reflectance.write_summary(summaries, campaign_filter)
    return summaries


def run_pipeline_for_file(meta: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    input_csv = Path(meta["path"])
    campaign = str(meta["campaign"])
    sensor = str(meta["sensor"])
    campaign_dir = args.insitu_root / campaign

    eligibility_summary = eligibility.process_file(input_csv, campaign, delimiter=args.delimiter)
    eligibility_csv = Path(eligibility_summary["output_csv"])

    hard_summary = hard_filter.process_file(
        eligibility_csv,
        campaign,
        sensor,
        args.delimiter,
        args.rrs_column_regex,
        args.wl_min,
        args.wl_max,
        args.required_wavelengths,
        args.max_wavelength_difference_nm,
        args.eo_input_root,
        args.radiometry_neighbour_threshold_fraction,
        args.radiometry_match_tolerance_seconds,
        args.radiometry_neighbour_max_interval_seconds,
    )
    hard_csv = Path(hard_summary["output_csv"])

    hard_meta = {"campaign": campaign, "sensor": sensor, "path": hard_csv, "campaign_dir": campaign_dir}
    bottom_summaries = run_bottom_reflectance_for_files(
        [hard_meta],
        args.delimiter,
        args.secchi_table,
    )
    bottom_csv = Path(bottom_summaries[0]["output_csv"])
    merge_summary = merge_bottom_into_hard_filter(hard_csv, bottom_csv, args.delimiter)

    flag_summary = flags.process_file(hard_meta, args.delimiter, args.transect_timeframes_csv)
    flags_csv = Path(flag_summary["output_csv"])
    final_csv = final_output_path_for(input_csv)
    shutil.copyfile(flags_csv, final_csv)

    return {
        "campaign": campaign,
        "sensor": sensor,
        "input_csv": str(input_csv),
        "eligibility_csv": str(eligibility_csv),
        "hard_filter_csv": str(hard_csv),
        "bottom_reflectance_csv": str(bottom_csv),
        "flags_csv": str(flags_csv),
        "final_qc_csv": str(final_csv),
        "final_retained": merge_summary["retained_after_bottom"],
    }


def parse_required_wavelengths(text: str) -> tuple[float, ...]:
    return hard_filter.parse_required_wavelengths(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run eligibility, hard filtering, bottom-reflectance screening, and flags.")
    parser.add_argument("--insitu-root", type=Path, default=INSITU_ROOT)
    parser.add_argument("--eo-input-root", type=Path, default=EO_INPUT_ROOT)
    parser.add_argument("--transect-timeframes-csv", type=Path, default=TRANSECT_TIMEFRAMES_CSV)
    parser.add_argument("--campaign", default=None)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument(
        "--current-only",
        action="store_true",
        help="Process only final_metadata files whose filename contains _current.",
    )
    parser.add_argument("--delimiter", default=None)
    parser.add_argument("--secchi-table", type=Path, default=None)
    parser.add_argument("--rrs-column-regex", default=hard_filter.DEFAULT_RRS_COLUMN_REGEX)
    parser.add_argument("--wl-min", type=float, default=hard_filter.DEFAULT_WL_MIN)
    parser.add_argument("--wl-max", type=float, default=hard_filter.DEFAULT_WL_MAX)
    parser.add_argument("--required-wavelengths", type=parse_required_wavelengths, default=hard_filter.DEFAULT_REQUIRED_WAVELENGTHS)
    parser.add_argument("--max-wavelength-difference-nm", type=float, default=hard_filter.DEFAULT_MAX_WAVELENGTH_DIFFERENCE_NM)
    parser.add_argument("--radiometry-neighbour-threshold-fraction", type=float, default=hard_filter.RADIOMETRY_NEIGHBOUR_THRESHOLD_FRACTION)
    parser.add_argument("--radiometry-match-tolerance-seconds", type=float, default=hard_filter.RADIOMETRY_MATCH_TOLERANCE_SECONDS)
    parser.add_argument("--radiometry-neighbour-max-interval-seconds", type=float, default=hard_filter.RADIOMETRY_NEIGHBOUR_MAX_INTERVAL_SECONDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eligibility.INSITU_ROOT = args.insitu_root
    hard_filter.INSITU_ROOT = args.insitu_root
    flags.INSITU_ROOT = args.insitu_root

    metadata_files = discover_metadata_files(args.insitu_root, args.campaign, args.input_csv, args.current_only)
    if not metadata_files:
        suffix = f" for campaign {args.campaign}" if args.campaign else ""
        raise SystemExit(f"No canonical hyperspectral metadata files found below {args.insitu_root}{suffix}")

    summaries = []
    errors = []
    for meta in metadata_files:
        try:
            summary = run_pipeline_for_file(meta, args)
            summaries.append(summary)
            print(f"Processed {summary['input_csv']} -> {summary['final_qc_csv']} ({summary['final_retained']} retained)")
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

    propagate_summary, propagate_errors = propagate_qc.run_batch(args.insitu_root, args.campaign)
    print(f"Propagated final QC to convolved/current files: {len(propagate_summary)}")
    if not propagate_errors.empty:
        print("Propagation errors:")
        print(propagate_errors.to_string(index=False))


if __name__ == "__main__":
    main()
