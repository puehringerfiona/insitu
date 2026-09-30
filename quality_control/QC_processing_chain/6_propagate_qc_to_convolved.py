from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd


INSITU_ROOT = Path(r"C:\Users\puehrifi\Documents\insitu")
SCRIPT_DIR = Path(__file__).resolve().parent
SUMMARY_CSV = SCRIPT_DIR / "convolved_qc_propagation_summary.csv"
ERRORS_CSV = SCRIPT_DIR / "convolved_qc_propagation_errors.csv"

JOIN_KEYS = ["fid_ramses", "timestamp (UTC)"]
CONVOLVED_SENSOR_RE = re.compile(
    r"^(?P<campaign>\d{8}_[A-Za-z0-9]+)_metadata_with_Rrs_conv_(?P<sensor>L8|L9|S2A|S2B)\.csv$"
)
FINAL_QC_RE = re.compile(
    r"^(?P<campaign>\d{8}_[A-Za-z0-9]+)_metadata_with_Rrs_(?P<sensor>L8|L9|S2A|S2B)_final_qc\.csv$"
)
SENSOR_TOKEN_RE = re.compile(r"_(?P<sensor>L8|L9|S2A|S2B)(?:_|\.csv$)", re.IGNORECASE)

QC_COLUMN_PREFIXES = (
    "eligibility_",
    "hard_filter_",
    "rrs_",
    "bottom_reflectance_",
    "metric_",
    "flag_",
)
QC_COLUMN_NAMES = {
    "satellite_overpass_utc",
    "eo_product",
    "temporal_offset_min",
    "secchi_depth_m",
    "kd_from_secchi_m-1",
    "flags_evaluated",
    "flag_count_without_hard_filter",
    "flag_count_incl_hard_filter",
}
QC_RAW_METRIC_RE = re.compile(r"^raw_(?:Ed|Ld|Lu|neighbour)_", re.IGNORECASE)


@dataclass
class CsvTable:
    path: Path
    sep: str
    df: pd.DataFrame


def read_csv_auto(path: Path, **kwargs) -> CsvTable:
    attempts = [
        {"sep": ";"},
        {"sep": ","},
        {"sep": "\t"},
        {"sep": None, "engine": "python"},
    ]
    for opts in attempts:
        try:
            df = pd.read_csv(path, low_memory=False, **opts, **kwargs)
            if len(df.columns) > 1:
                df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)
                sep = opts.get("sep")
                if sep is None:
                    sep = ";"
                return CsvTable(path=path, sep=sep, df=df)
        except Exception:
            continue
    raise ValueError(f"Could not parse CSV: {path}")


def is_campaign_dir(path: Path) -> bool:
    return path.is_dir() and bool(re.fullmatch(r"\d{8}_[A-Za-z0-9]+", path.name))


def normalize_join_keys(df: pd.DataFrame, source: str) -> pd.DataFrame:
    missing = [key for key in JOIN_KEYS if key not in df.columns]
    if missing:
        raise ValueError(f"{source} missing join key columns: {missing}")
    out = df.copy()
    for key in JOIN_KEYS:
        numeric = pd.to_numeric(out[key], errors="coerce")
        out[f"__join_{key}"] = np.where(
            numeric.notna(),
            numeric.round().astype("Int64").astype(str),
            out[key].astype(str).str.strip(),
        )
        out.loc[out[f"__join_{key}"].isin(["<NA>", "nan", "NaN", "None"]), f"__join_{key}"] = ""
    return out


def join_key_columns() -> list[str]:
    return [f"__join_{key}" for key in JOIN_KEYS]


def score_file(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    score = 0
    if "_metadata_with_rrs_" in name:
        score += 20
    if "_conv_" in name:
        score -= 5
    if "_current" in name:
        score -= 20
    if "_example" in name:
        score -= 20
    if "_old" in name:
        score -= 30
    if any(part.lower() == "obsolete" for part in path.parts):
        score -= 100
    return score, path.name


def find_target_metadata_files(campaign_dir: Path) -> list[Path]:
    final_metadata = campaign_dir / "final_metadata"
    if not final_metadata.exists():
        return []
    files = []
    for path in final_metadata.glob(f"{campaign_dir.name}_metadata_with_Rrs_*.csv"):
        name = path.name
        if "_conv_" in name or "_current" in name:
            files.append(path)
    return sorted(files, key=score_file)


def infer_sensor(path: Path) -> str | None:
    convolved_match = CONVOLVED_SENSOR_RE.match(path.name)
    if convolved_match:
        return convolved_match.group("sensor").upper()
    match = SENSOR_TOKEN_RE.search(path.name)
    return match.group("sensor").upper() if match else None


def find_final_qc(campaign_dir: Path, sensor: str) -> Path | None:
    qc_dir = campaign_dir / "visualization_analysis" / "quality_control"
    pattern = f"{campaign_dir.name}_metadata_with_Rrs_{sensor}_final_qc.csv"
    matches = [
        path
        for path in qc_dir.glob(pattern)
        if "obsolete" not in [part.lower() for part in path.parts]
    ]
    return sorted(matches, key=score_file)[-1] if matches else None


def validate_unique_keys(df: pd.DataFrame, label: str) -> None:
    key_cols = join_key_columns()
    duplicates = df.duplicated(key_cols, keep=False)
    if duplicates.any():
        duplicate_count = int(duplicates.sum())
        raise ValueError(f"{label} has {duplicate_count} rows with duplicated join keys {JOIN_KEYS}")


def is_qc_column(column: str) -> bool:
    if column in JOIN_KEYS or column.startswith("__join_"):
        return False
    return (
        column in QC_COLUMN_NAMES
        or column.startswith(QC_COLUMN_PREFIXES)
        or bool(QC_RAW_METRIC_RE.match(column))
    )


def build_final_qc_flags(final_qc_df: pd.DataFrame) -> pd.DataFrame:
    qc = normalize_join_keys(final_qc_df, "final QC")
    validate_unique_keys(qc, "final QC")
    qc_columns = [column for column in qc.columns if is_qc_column(column)]
    if not qc_columns:
        raise ValueError("final QC has no recognized QC columns to propagate.")
    return qc[join_key_columns() + qc_columns].copy()


def remove_existing_qc_columns(df: pd.DataFrame, propagated_columns: list[str]) -> pd.DataFrame:
    removable = [column for column in propagated_columns if column in df.columns]
    return df.drop(columns=removable) if removable else df


def retained_series(result: pd.DataFrame) -> pd.Series:
    if "hard_filter_retain" in result.columns:
        return result["hard_filter_retain"].astype("boolean").fillna(False)
    if "hard_filter_exclude" in result.columns:
        return ~result["hard_filter_exclude"].astype("boolean").fillna(True)
    return pd.Series(False, index=result.index, dtype=bool)


def propagate_one(metadata_csv: Path, final_qc: Path, output_dir: Path) -> dict[str, object]:
    campaign = metadata_csv.name.split("_metadata_with_Rrs_")[0]
    sensor = infer_sensor(metadata_csv)
    if sensor is None:
        raise ValueError(f"Could not infer sensor from filename: {metadata_csv.name}")

    metadata_table = read_csv_auto(metadata_csv)
    metadata = normalize_join_keys(metadata_table.df, "target metadata")
    validate_unique_keys(metadata, "target metadata")

    qc_flags = build_final_qc_flags(read_csv_auto(final_qc).df)
    propagated_columns = [column for column in qc_flags.columns if not column.startswith("__join_")]
    metadata = remove_existing_qc_columns(metadata, propagated_columns)

    result = metadata.merge(qc_flags, on=join_key_columns(), how="left", validate="one_to_one")
    missing_qc_match = result[propagated_columns].isna().all(axis=1)
    result["qc_propagation_missing_final_qc_match"] = missing_qc_match
    result["propagated_final_qc_source"] = str(final_qc)

    helper_cols = join_key_columns()
    result = result.drop(columns=[col for col in helper_cols if col in result.columns])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_qc = output_dir / f"{metadata_csv.stem}_qc.csv"
    result.to_csv(output_qc, sep=metadata_table.sep, index=False)

    retained = retained_series(result)
    return {
        "campaign": campaign,
        "sensor": sensor,
        "status": "ok",
        "input_metadata": str(metadata_csv),
        "final_qc": str(final_qc),
        "output_qc": str(output_qc),
        "rows": int(len(result)),
        "retained": int(retained.sum()),
        "excluded": int((~retained).sum()),
        "missing_final_qc_match": int(missing_qc_match.sum()),
        "propagated_qc_columns": len(propagated_columns),
    }


def get_campaign_dirs(root: Path, campaign_filter: str | None = None) -> list[Path]:
    if campaign_filter:
        path = root / campaign_filter
        return [path] if is_campaign_dir(path) else []
    return sorted(path for path in root.iterdir() if is_campaign_dir(path))


def run_batch(root: Path, campaign_filter: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    errors = []
    for campaign_dir in get_campaign_dirs(root, campaign_filter):
        output_dir = campaign_dir / "visualization_analysis" / "quality_control"
        for metadata_csv in find_target_metadata_files(campaign_dir):
            sensor = infer_sensor(metadata_csv)
            if sensor is None:
                errors.append({
                    "campaign": campaign_dir.name,
                    "sensor": "",
                    "input_metadata": str(metadata_csv),
                    "error": "could not infer sensor from filename",
                })
                continue
            final_qc = find_final_qc(campaign_dir, sensor)
            if final_qc is None:
                errors.append({
                    "campaign": campaign_dir.name,
                    "sensor": sensor,
                    "input_metadata": str(metadata_csv),
                    "error": "missing final QC",
                })
                continue
            try:
                summaries.append(propagate_one(metadata_csv, final_qc, output_dir))
            except Exception as exc:
                errors.append({
                    "campaign": campaign_dir.name,
                    "sensor": sensor,
                    "input_metadata": str(metadata_csv),
                    "final_qc": str(final_qc),
                    "error": str(exc),
                })

    summary_df = pd.DataFrame(summaries)
    errors_df = pd.DataFrame(
        errors,
        columns=["campaign", "sensor", "input_metadata", "final_qc", "error"],
    )
    summary_df.to_csv(SUMMARY_CSV, index=False)
    errors_df.to_csv(ERRORS_CSV, index=False)
    return summary_df, errors_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Propagate final QC columns to convolved and current final-metadata files."
    )
    parser.add_argument("--insitu-root", type=Path, default=INSITU_ROOT)
    parser.add_argument("--campaign", default=None, help="Optional single campaign, e.g. 20230610_CST.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_df, errors_df = run_batch(args.insitu_root, args.campaign)
    print(f"Wrote summary: {SUMMARY_CSV}")
    print(f"Wrote errors: {ERRORS_CSV}")
    if not summary_df.empty:
        print(summary_df[["campaign", "sensor", "rows", "retained", "excluded", "missing_final_qc_match"]].to_string(index=False))
    if not errors_df.empty:
        print("Errors:")
        print(errors_df.to_string(index=False))


if __name__ == "__main__":
    main()
