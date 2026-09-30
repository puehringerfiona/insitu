"""
Mandatory hard exclusion filters for in situ Rrs metadata tables.

This stage consumes the eligibility QC products and adds non-compensable hard
exclusion flags for acquisition geometry, spectral validity, required
satellite-comparison Rrs bands, raw radiometry consistency, and temporal
matchup offset.
"""

from __future__ import annotations

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


INSITU_ROOT = Path(r"C:\Users\puehrifi\Documents\insitu")
EO_INPUT_ROOT = Path(r"C:\Users\puehrifi\Documents\eo_data\input")

ELIGIBILITY_PATTERN = re.compile(
    r"^(?P<campaign>\d{8}_[A-Za-z0-9]+)_metadata_with_Rrs_(?P<sensor>L8|L9|S2A|S2B)(?:_[A-Za-z0-9]+)*_eligibility_qc\.csv$",
    re.IGNORECASE,
)

DEFAULT_RRS_COLUMN_REGEX = r"rrs|^Rrs_|^\d{3}(?:\.\d+)?$"
DEFAULT_WL_MIN = 350.0
DEFAULT_WL_MAX = 900.0
DEFAULT_REQUIRED_WAVELENGTHS = (443.0, 560.0, 665.0)
DEFAULT_MAX_WAVELENGTH_DIFFERENCE_NM = 2.0

VAZ_ACCEPT_LOW = 75.0
VAZ_ACCEPT_HIGH = 150.0
TEMPORAL_MAX_OFFSET_MIN = 180.0

RADIOMETRY_NEIGHBOUR_THRESHOLD_FRACTION = 0.25
RADIOMETRY_MATCH_TOLERANCE_SECONDS = 30.0
RADIOMETRY_NEIGHBOUR_MAX_INTERVAL_SECONDS = 180.0
RADIOMETRY_MEDIAN_NEIGHBOUR_THRESHOLD_FRACTION = 0.25
RADIOMETRY_MEDIAN_NEIGHBOUR_TARGET_WAVELENGTH_NM = 560.0
RADIOMETRY_MEDIAN_NEIGHBOUR_MAX_WAVELENGTH_DIFFERENCE_NM = 2.0
RADIOMETRY_MEDIAN_NEIGHBOUR_MAX_INTERVAL_SECONDS = 180.0
RADIOMETRY_NESTED_MEDIAN_NEIGHBOUR_THRESHOLDS = {
    "25pct": RADIOMETRY_MEDIAN_NEIGHBOUR_THRESHOLD_FRACTION,
}
NIR_MIN_NM = 765.0
NIR_MAX_NM = 900.0
BLUE_MIN_NM = 350.0
BLUE_MAX_NM = 450.0
MIN_TOTAL_NEGATIVE_COUNT = 20
MIN_BLUE_NEGATIVE_COUNT = 20
NIR_SLOPE_THRESHOLD = -8.75e-7
NIR_NEGATIVE_FRACTION_WITH_SLOPE = 0.50
NIR_NEGATIVE_FRACTION_DIRECT = 0.70
POSITIVE_BASELINE_MIN_OVER_MEDIAN_THRESHOLD = 0.60
ISO_DATE_RE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}")

# Columns computed for internal use (combining into a kept column, or as an
# input another kept column already summarises) but not written to the output
# CSV. Each entry below is annotated with what still carries its information.
OUTPUT_DROP_COLUMNS = [
    # Denominator/count/weaker-threshold inputs to kept diagnostics; never
    # themselves compared against a QC threshold.
    "rrs_finite_band_count",  # denominator of the kept rrs_negative_fraction
    "rrs_has_any_negative",  # weaker than the kept rrs_negative_count >= 20 gate
    "rrs_nir_negative_count_765_900",  # feeds the kept rrs_nir_negative_fraction_765_900
    # Availability flags for the nested-median-560 check: default-False exactly
    # like "evaluated and fine", so they carry no decision-relevant information
    # beyond the kept continuous relative_difference columns.
    "raw_Ed_nested_median_neighbour_560_180s_25pct_radiometry_available",
    "raw_Ld_nested_median_neighbour_560_180s_25pct_radiometry_available",
    "raw_Ed_nested_median_neighbour_560_transect_25pct_radiometry_available",
    "raw_Ld_nested_median_neighbour_560_transect_25pct_radiometry_available",
    # Raw point values/wavelengths at 560 nm: only their nonfinite verdict
    # (combined into hard_filter_raw_radiometry_560_nonfinite, kept) is acted on.
    "raw_Ed_560_finite_available", "raw_Ed_560_wavelength_nm", "raw_Ed_560_value",
    "hard_filter_raw_Ed_560_nonfinite",
    "raw_Ld_560_finite_available", "raw_Ld_560_wavelength_nm", "raw_Ld_560_value",
    "hard_filter_raw_Ld_560_nonfinite",
    "raw_Lu_560_finite_available", "raw_Lu_560_wavelength_nm", "raw_Lu_560_value",
    "hard_filter_raw_Lu_560_nonfinite",
    # Ed-OR-Ld combination (without the nonfinite branch): redundant now that
    # both per-quantity difference_gt_25pct booleans are kept directly.
    "hard_filter_raw_radiometry_nested_median_neighbour_560_180s_difference_gt_25pct",
    "hard_filter_raw_radiometry_nested_median_neighbour_560_transect_difference_gt_25pct",
    # Exact duplicates: result["hard_filter_exclude"] etc. are literal copies of
    # these (see the nested_180s/nested_transect aliasing below), and every
    # consumer uses the plain/_strict names.
    "hard_filter_exclusion_reason_nested_180s", "hard_filter_exclude_nested_180s",
    "hard_filter_retain_nested_180s",
    "hard_filter_exclusion_reason_nested_transect", "hard_filter_exclude_nested_transect",
    "hard_filter_retain_nested_transect",
    # Exact negation of hard_filter_retain_strict (kept), with no consumers.
    "hard_filter_exclude_strict",
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


def append_reason(current: pd.Series, mask: pd.Series | np.ndarray, reason: str) -> pd.Series:
    out = current.copy()
    mask = pd.Series(mask, index=out.index).fillna(False)
    out.loc[mask] = out.loc[mask].map(lambda old: f"{old};{reason}" if old else reason)
    return out


def parse_datetime_mixed(series: pd.Series, utc: bool = True) -> pd.Series:
    text = series.astype("string").str.strip()
    valid = text.notna() & (text != "") & (text.str.lower() != "nan")
    iso = text.str.match(ISO_DATE_RE, na=False) & valid
    other = valid & ~iso
    dtype = "datetime64[ns, UTC]" if utc else "datetime64[ns]"
    parsed = pd.Series(pd.NaT, index=series.index, dtype=dtype)
    if iso.any():
        parsed.loc[iso] = pd.to_datetime(text.loc[iso], errors="coerce", utc=utc, dayfirst=False)
    if other.any():
        parsed.loc[other] = pd.to_datetime(text.loc[other], errors="coerce", utc=utc, dayfirst=True)
    return parsed


def parse_wavelength_from_column(column: object, prefix_regex: str | None) -> float | None:
    text = str(column).strip()
    if prefix_regex and re.search(prefix_regex, text, flags=re.IGNORECASE) is None:
        return None
    matches = re.findall(r"(?<!\d)(\d{3}(?:\.\d+)?)(?!\d)", text)
    return float(matches[-1]) if matches else None


def find_rrs_columns(
    df: pd.DataFrame,
    wl_min: float,
    wl_max: float,
    rrs_column_regex: str | None,
) -> list[tuple[str, float]]:
    columns = []
    for col in df.columns:
        wl = parse_wavelength_from_column(col, rrs_column_regex)
        if wl is not None and wl_min <= wl <= wl_max:
            columns.append((str(col), wl))
    columns = sorted(columns, key=lambda item: item[1])
    if not columns:
        raise ValueError("No Rrs wavelength columns found.")
    return columns


def resolve_required_wavelength_columns(
    rrs_columns: list[tuple[str, float]],
    required_wavelengths: Iterable[float],
    max_difference_nm: float,
) -> dict[float, str]:
    resolved = {}
    for target in required_wavelengths:
        selected_col, selected_wl = min(rrs_columns, key=lambda item: abs(item[1] - target))
        difference = abs(selected_wl - target)
        if difference > max_difference_nm:
            raise ValueError(
                f"No Rrs column within {max_difference_nm:g} nm of {target:g} nm. "
                f"Nearest is {selected_col} ({selected_wl:g} nm), {difference:g} nm away."
            )
        resolved[target] = selected_col
    return resolved


def columns_in_range(rrs_columns: list[tuple[str, float]], wl_min: float, wl_max: float) -> list[str]:
    return [col for col, wl in rrs_columns if wl_min <= wl <= wl_max]


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return np.nan
    return float(np.polyfit(x[finite], y[finite], 1)[0])


def compute_geometry_qc(df: pd.DataFrame) -> pd.DataFrame:
    qc = pd.DataFrame(index=df.index)
    if "relative azimuth angle" in df.columns:
        vaz = to_numeric(df["relative azimuth angle"])
    else:
        vaz = pd.Series(np.nan, index=df.index)
    qc["hard_filter_vaz_deg"] = vaz
    qc["hard_filter_vaz_outside_75_150"] = ~(np.isfinite(vaz) & vaz.between(VAZ_ACCEPT_LOW, VAZ_ACCEPT_HIGH))
    return qc


def compute_spectral_qc(
    df: pd.DataFrame,
    rrs_columns: list[tuple[str, float]],
    required_columns: dict[float, str],
) -> pd.DataFrame:
    wl_cols = [col for col, _ in rrs_columns]
    wavelengths = np.asarray([wl for _, wl in rrs_columns], dtype=float)
    spectra = df[wl_cols].apply(to_numeric).to_numpy(dtype=float)

    negative = spectra < 0
    finite = np.isfinite(spectra)
    total_negative_count = negative.sum(axis=1)
    finite_count = finite.sum(axis=1)
    negative_fraction = np.divide(
        total_negative_count,
        finite_count,
        out=np.zeros_like(total_negative_count, dtype=float),
        where=finite_count > 0,
    )

    nir_cols = columns_in_range(rrs_columns, NIR_MIN_NM, NIR_MAX_NM)
    blue_cols = columns_in_range(rrs_columns, BLUE_MIN_NM, BLUE_MAX_NM)

    if nir_cols:
        nir_wavelengths = np.asarray([wl for col, wl in rrs_columns if col in nir_cols], dtype=float)
        nir_values = df[nir_cols].apply(to_numeric).to_numpy(dtype=float)
        nir_negative = nir_values < 0
        nir_finite = np.isfinite(nir_values)
        nir_negative_count = nir_negative.sum(axis=1)
        nir_finite_count = nir_finite.sum(axis=1)
        nir_negative_fraction = np.divide(
            nir_negative_count,
            nir_finite_count,
            out=np.full_like(nir_negative_count, np.nan, dtype=float),
            where=nir_finite_count > 0,
        )
        nir_slope = np.asarray([linear_slope(nir_wavelengths, row) for row in nir_values], dtype=float)
    else:
        nir_negative_count = np.zeros(len(df), dtype=int)
        nir_negative_fraction = np.full(len(df), np.nan, dtype=float)
        nir_slope = np.full(len(df), np.nan, dtype=float)

    if blue_cols:
        blue_values = df[blue_cols].apply(to_numeric).to_numpy(dtype=float)
        blue_negative_count = (blue_values < 0).sum(axis=1)
    else:
        blue_negative_count = np.zeros(len(df), dtype=int)

    required_values = df[list(required_columns.values())].apply(to_numeric)
    required_nonfinite = ~np.isfinite(required_values).all(axis=1)
    required_nonpositive = (required_values <= 0).any(axis=1)

    negative_baseline_shift = (
        total_negative_count >= MIN_TOTAL_NEGATIVE_COUNT
    ) & (
        ((nir_slope < NIR_SLOPE_THRESHOLD) & (nir_negative_fraction > NIR_NEGATIVE_FRACTION_WITH_SLOPE))
        | (nir_negative_fraction > NIR_NEGATIVE_FRACTION_DIRECT)
        | (blue_negative_count >= MIN_BLUE_NEGATIVE_COUNT)
    )

    spectra_min = np.nanmin(spectra, axis=1)
    spectra_median = np.nanmedian(spectra, axis=1)
    min_over_median = np.divide(
        spectra_min,
        spectra_median,
        out=np.full_like(spectra_min, np.nan, dtype=float),
        where=np.isfinite(spectra_median) & (spectra_median != 0),
    )
    positive_baseline_shift = (
        np.isfinite(min_over_median)
        & np.isfinite(spectra_median)
        & (spectra_median > 0)
        & (min_over_median > POSITIVE_BASELINE_MIN_OVER_MEDIAN_THRESHOLD)
    )

    qc = pd.DataFrame(index=df.index)
    qc["rrs_finite_band_count"] = finite_count
    qc["rrs_negative_count"] = total_negative_count
    qc["rrs_negative_fraction"] = negative_fraction
    qc["rrs_has_any_negative"] = total_negative_count > 0
    qc["rrs_nir_negative_count_765_900"] = nir_negative_count
    qc["rrs_nir_negative_fraction_765_900"] = nir_negative_fraction
    qc["rrs_nir_slope_765_900"] = nir_slope
    qc["rrs_blue_negative_count_350_450"] = blue_negative_count
    qc["rrs_min_over_median"] = min_over_median
    qc["rrs_required_nonfinite"] = required_nonfinite
    qc["rrs_required_nonpositive"] = required_nonpositive
    qc["rrs_negative_baseline_shift"] = negative_baseline_shift
    qc["rrs_positive_baseline_shift"] = positive_baseline_shift
    return qc


def parse_metadata_times(df: pd.DataFrame) -> pd.Series:
    if "timestamp (UTC)" in df.columns:
        numeric = to_numeric(df["timestamp (UTC)"])
        return pd.to_datetime(numeric, unit="s", errors="coerce", utc=True)
    if "datetime (UTC)" in df.columns:
        return parse_datetime_mixed(df["datetime (UTC)"], utc=True)
    if "datetime (CET)" in df.columns:
        return parse_datetime_mixed(df["datetime (CET)"], utc=True)
    raise ValueError("No datetime (UTC), timestamp (UTC), or datetime (CET) column found.")


def sensor_from_product_path(path: Path) -> str | None:
    name = path.name.upper()
    if name.startswith("LC08") or "LANDSAT_8" in name:
        return "L8"
    if name.startswith("LC09") or "LANDSAT_9" in name:
        return "L9"
    if name.startswith("S2A"):
        return "S2A"
    if name.startswith("S2B"):
        return "S2B"
    return None


def parse_landsat_mtl_time(path: Path) -> pd.Timestamp | None:
    date_acquired = None
    scene_center_time = None
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "DATE_ACQUIRED" in line:
                date_acquired = line.split("=", 1)[1].strip().strip('"')
            elif "SCENE_CENTER_TIME" in line:
                scene_center_time = line.split("=", 1)[1].strip().strip('"')
    if not date_acquired or not scene_center_time:
        return None
    return pd.to_datetime(f"{date_acquired}T{scene_center_time}", utc=True)


def parse_sentinel_product_time(path: Path) -> pd.Timestamp | None:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None
    for elem in root.iter():
        if elem.tag.split("}")[-1] == "PRODUCT_START_TIME" and elem.text:
            return pd.to_datetime(elem.text.strip(), utc=True)
    return None


def find_overpass(eo_input_root: Path, campaign: str, sensor: str) -> dict[str, object] | None:
    campaign_root = eo_input_root / campaign
    if not campaign_root.exists():
        return None

    overpasses = []
    for mtl_path in campaign_root.rglob("*_MTL.txt"):
        product_sensor = sensor_from_product_path(mtl_path.parent)
        timestamp = parse_landsat_mtl_time(mtl_path)
        if product_sensor == sensor and timestamp is not None:
            overpasses.append({"time": timestamp, "metadata": str(mtl_path), "product": str(mtl_path.parent)})

    for xml_path in campaign_root.rglob("MTD_MSIL1C.xml"):
        safe_dir = next((parent for parent in xml_path.parents if parent.suffix.upper() == ".SAFE"), xml_path.parent)
        product_sensor = sensor_from_product_path(safe_dir)
        timestamp = parse_sentinel_product_time(xml_path)
        if product_sensor == sensor and timestamp is not None:
            overpasses.append({"time": timestamp, "metadata": str(xml_path), "product": str(safe_dir)})

    return sorted(overpasses, key=lambda item: item["product"])[0] if overpasses else None


def compute_temporal_qc(df: pd.DataFrame, campaign: str, sensor: str, eo_input_root: Path) -> pd.DataFrame:
    qc = pd.DataFrame(index=df.index)
    overpass = find_overpass(eo_input_root, campaign, sensor)
    if overpass is None:
        qc["satellite_overpass_utc"] = pd.NaT
        qc["eo_product"] = pd.NA
        qc["temporal_offset_min"] = np.nan
        qc["hard_filter_temporal_offset_gt_180min"] = True
        qc["hard_filter_missing_satellite_overpass"] = True
        return qc

    metadata_times = parse_metadata_times(df)
    offset = (metadata_times - overpass["time"]).abs().dt.total_seconds() / 60.0
    qc["satellite_overpass_utc"] = overpass["time"]
    qc["eo_product"] = overpass["product"]
    qc["temporal_offset_min"] = offset
    qc["hard_filter_temporal_offset_gt_180min"] = offset > TEMPORAL_MAX_OFFSET_MIN
    qc["hard_filter_missing_satellite_overpass"] = False
    return qc


def local_naive_times(df: pd.DataFrame) -> pd.Series:
    if "datetime (CET)" in df.columns:
        return parse_datetime_mixed(df["datetime (CET)"], utc=True).dt.tz_convert("Europe/Zurich").dt.tz_localize(None)
    return parse_metadata_times(df).dt.tz_convert("Europe/Zurich").dt.tz_localize(None)


def read_spectrum(path: Path) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    spectrum = pd.read_csv(path, sep=None, engine="python", index_col=0)
    spectrum.index = pd.to_numeric(spectrum.index, errors="coerce")
    spectrum = spectrum.loc[spectrum.index.notna()].sort_index()
    sample_times = parse_datetime_mixed(pd.Series(spectrum.columns), utc=False)
    valid_cols = [col for col, ts in zip(spectrum.columns, sample_times) if pd.notna(ts)]
    spectrum = spectrum[valid_cols].apply(pd.to_numeric, errors="coerce")
    sample_times = pd.DatetimeIndex(parse_datetime_mixed(pd.Series(valid_cols), utc=False))
    order = np.argsort(sample_times.values)
    return spectrum.iloc[:, order], sample_times[order]


def nearest_metadata_index(sample_times: pd.DatetimeIndex, metadata_times: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    meta = pd.DataFrame({"metadata_time": metadata_times}).dropna().sort_values("metadata_time")
    if meta.empty:
        return np.full(len(sample_times), -1, dtype=int), np.full(len(sample_times), np.nan, dtype=float)
    matched = pd.merge_asof(
        pd.DataFrame({"sample_time": sample_times}).sort_values("sample_time"),
        meta.reset_index().rename(columns={"index": "metadata_index"}),
        left_on="sample_time",
        right_on="metadata_time",
        direction="nearest",
    )
    delta = (matched["sample_time"] - matched["metadata_time"]).abs().dt.total_seconds().to_numpy(dtype=float)
    return matched["metadata_index"].fillna(-1).to_numpy(dtype=int), delta


def find_radiometry_spectrum(campaign_dir: Path, campaign: str, quantity: str) -> Path | None:
    spectra_dir = campaign_dir / "preprocessing" / f"{campaign}_ramses_spectra"
    candidates = sorted(spectra_dir.glob(f"{campaign}_*_{quantity}.csv")) if spectra_dir.exists() else []
    return candidates[0] if candidates else None


def find_active_neighbour_index(index: int, active: np.ndarray, step: int) -> int | None:
    current = index + step
    while 0 <= current < len(active):
        if active[current]:
            return current
        current += step
    return None


def neighbour_reference(
    values: np.ndarray,
    index: int,
    sample_times: pd.DatetimeIndex,
    max_interval_seconds: float,
    active: np.ndarray | None = None,
) -> np.ndarray | None:
    if active is None:
        active = np.ones(values.shape[1], dtype=bool)

    neighbours = []
    previous_index = find_active_neighbour_index(index, active, -1)
    if previous_index is not None:
        delta = abs((sample_times[index] - sample_times[previous_index]).total_seconds())
        if delta <= max_interval_seconds:
            neighbours.append(values[:, previous_index])

    next_index = find_active_neighbour_index(index, active, 1)
    if next_index is not None:
        delta = abs((sample_times[next_index] - sample_times[index]).total_seconds())
        if delta <= max_interval_seconds:
            neighbours.append(values[:, next_index])

    if not neighbours:
        return None
    return np.nanmean(np.column_stack(neighbours), axis=1)


def compute_neighbour_relative_difference(
    spectrum: pd.DataFrame,
    sample_times: pd.DatetimeIndex,
    max_interval_seconds: float,
    threshold_fraction: float,
    active: np.ndarray | None = None,
) -> np.ndarray:
    values = spectrum.to_numpy(dtype=float)
    if active is None:
        active = np.ones(values.shape[1], dtype=bool)

    median_differences = np.full(values.shape[1], np.nan, dtype=float)
    for idx in range(values.shape[1]):
        if not active[idx]:
            continue
        reference = neighbour_reference(values, idx, sample_times, max_interval_seconds, active)
        if reference is None:
            continue
        valid = np.isfinite(values[:, idx]) & np.isfinite(reference) & (np.abs(reference) > 0)
        if valid.any():
            relative_difference = np.abs(values[valid, idx] - reference[valid]) / np.abs(reference[valid])
            median_differences[idx] = np.nanmedian(relative_difference)
    return median_differences


def resolve_nearest_wavelength_index(
    wavelengths: np.ndarray,
    target_wavelength_nm: float,
    max_difference_nm: float,
) -> int | None:
    if wavelengths.size == 0:
        return None
    idx = int(np.nanargmin(np.abs(wavelengths - target_wavelength_nm)))
    return idx if abs(float(wavelengths[idx]) - target_wavelength_nm) <= max_difference_nm else None


def find_active_time_neighbour_indices(
    index: int,
    active: np.ndarray,
    sample_times: pd.DatetimeIndex,
    max_interval_seconds: float,
) -> list[int]:
    center_time = sample_times[index]
    if pd.isna(center_time):
        return []

    neighbour_indices: list[int] = []
    for candidate_idx, candidate_time in enumerate(sample_times):
        if candidate_idx == index or not active[candidate_idx] or pd.isna(candidate_time):
            continue
        delta_seconds = abs((candidate_time - center_time).total_seconds())
        if delta_seconds <= max_interval_seconds:
            neighbour_indices.append(candidate_idx)
    return neighbour_indices


def neighbour_reference_at_wavelength(
    values_at_wavelength: np.ndarray,
    index: int,
    active: np.ndarray,
    sample_times: pd.DatetimeIndex,
    max_interval_seconds: float,
) -> float:
    neighbour_indices = find_active_time_neighbour_indices(
        index,
        active,
        sample_times,
        max_interval_seconds,
    )
    finite_neighbours = np.asarray(values_at_wavelength[neighbour_indices], dtype=float)
    finite_neighbours = finite_neighbours[np.isfinite(finite_neighbours)]
    if finite_neighbours.size == 0:
        return np.nan
    return float(np.nanmedian(finite_neighbours))


def compute_median_neighbour_relative_difference_at_wavelength(
    spectrum: pd.DataFrame,
    sample_times: pd.DatetimeIndex,
    max_interval_seconds: float,
    target_wavelength_nm: float,
    max_wavelength_difference_nm: float,
    active: np.ndarray | None = None,
) -> tuple[np.ndarray, float | None]:
    wavelengths = spectrum.index.to_numpy(dtype=float)
    target_idx = resolve_nearest_wavelength_index(
        wavelengths,
        target_wavelength_nm,
        max_wavelength_difference_nm,
    )
    if target_idx is None:
        return np.full(spectrum.shape[1], np.nan, dtype=float), None

    values_at_wavelength = spectrum.to_numpy(dtype=float)[target_idx, :]
    if active is None:
        active = np.ones(values_at_wavelength.shape[0], dtype=bool)

    differences = np.full(values_at_wavelength.shape[0], np.nan, dtype=float)
    for idx in range(values_at_wavelength.shape[0]):
        if not active[idx]:
            continue
        reference = neighbour_reference_at_wavelength(
            values_at_wavelength,
            idx,
            active,
            sample_times,
            max_interval_seconds,
        )
        value = values_at_wavelength[idx]
        if np.isfinite(value) and np.isfinite(reference) and abs(reference) > 0:
            differences[idx] = abs(value - reference) / abs(reference)
    return differences, float(wavelengths[target_idx])


def update_with_larger_finite(target: np.ndarray, current: np.ndarray) -> None:
    update = np.isfinite(current) & (~np.isfinite(target) | (current > target))
    target[update] = current[update]


def iterative_neighbour_relative_difference(
    spectrum: pd.DataFrame,
    sample_times: pd.DatetimeIndex,
    max_interval_seconds: float,
    threshold_fraction: float,
    max_iterations: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = spectrum.shape[1]
    active = np.ones(n_samples, dtype=bool)
    flagged = np.zeros(n_samples, dtype=bool)
    best_median = np.full(n_samples, np.nan, dtype=float)

    for _ in range(max_iterations):
        median = compute_neighbour_relative_difference(
            spectrum,
            sample_times,
            max_interval_seconds,
            threshold_fraction,
            active,
        )
        update_with_larger_finite(best_median, median)

        current_bad = active & np.isfinite(median) & (median > threshold_fraction)
        if not current_bad.any():
            break

        flagged |= current_bad
        active[current_bad] = False

    return best_median, flagged


def iterative_median_neighbour_relative_difference_at_wavelength(
    spectrum: pd.DataFrame,
    sample_times: pd.DatetimeIndex,
    max_interval_seconds: float,
    threshold_fraction: float,
    target_wavelength_nm: float,
    max_wavelength_difference_nm: float,
    max_iterations: int = 10,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    n_samples = spectrum.shape[1]
    active = np.ones(n_samples, dtype=bool)
    flagged = np.zeros(n_samples, dtype=bool)
    best_difference = np.full(n_samples, np.nan, dtype=float)
    resolved_wavelength = None

    for _ in range(max_iterations):
        differences, resolved_wavelength = compute_median_neighbour_relative_difference_at_wavelength(
            spectrum,
            sample_times,
            max_interval_seconds,
            target_wavelength_nm,
            max_wavelength_difference_nm,
            active,
        )
        update_with_larger_finite(best_difference, differences)

        current_bad = active & np.isfinite(differences) & (differences > threshold_fraction)
        if not current_bad.any():
            break

        flagged |= current_bad
        active[current_bad] = False

    return best_difference, flagged, resolved_wavelength


def nested_iterative_median_neighbour_relative_difference_at_wavelength(
    spectrum: pd.DataFrame,
    sample_times: pd.DatetimeIndex,
    max_interval_seconds: float,
    threshold_fractions: dict[str, float],
    target_wavelength_nm: float,
    max_wavelength_difference_nm: float,
    max_iterations: int = 100,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float | None]:
    n_samples = spectrum.shape[1]
    active = np.ones(n_samples, dtype=bool)
    cumulative_flagged = np.zeros(n_samples, dtype=bool)
    previous_best = np.full(n_samples, np.nan, dtype=float)
    differences_by_threshold: dict[str, np.ndarray] = {}
    flags_by_threshold: dict[str, np.ndarray] = {}
    resolved_wavelength = None

    for threshold_label, threshold_fraction in threshold_fractions.items():
        best_difference = previous_best.copy()
        local_active = active.copy()
        newly_flagged = np.zeros(n_samples, dtype=bool)

        for _ in range(max_iterations):
            differences, resolved_wavelength = compute_median_neighbour_relative_difference_at_wavelength(
                spectrum,
                sample_times,
                max_interval_seconds,
                target_wavelength_nm,
                max_wavelength_difference_nm,
                local_active,
            )
            update_with_larger_finite(best_difference, differences)

            current_bad = local_active & np.isfinite(differences) & (differences > threshold_fraction)
            if not current_bad.any():
                break

            newly_flagged |= current_bad
            local_active[current_bad] = False

        cumulative_flagged |= newly_flagged
        active &= ~newly_flagged
        previous_best = best_difference.copy()
        differences_by_threshold[threshold_label] = best_difference
        flags_by_threshold[threshold_label] = cumulative_flagged.copy()

    return differences_by_threshold, flags_by_threshold, resolved_wavelength


def compute_quantity_nested_median_neighbour_560_radiometry_qc(
    df: pd.DataFrame,
    campaign_dir: Path,
    campaign: str,
    quantity: str,
    variant_label: str,
    threshold_fractions: dict[str, float],
    match_tolerance_seconds: float,
    max_interval_seconds: float,
    target_wavelength_nm: float,
    max_wavelength_difference_nm: float,
) -> pd.DataFrame:
    # The resolved 560 nm channel is a fixed property of the instrument's wavelength
    # grid, not of the neighbourhood variant, so it is reported once via
    # raw_{quantity}_560_wavelength_nm (compute_quantity_560_finite_radiometry_qc)
    # rather than repeated here per variant/threshold.
    qc = pd.DataFrame(index=df.index)
    for threshold_label in threshold_fractions:
        qc[f"raw_{quantity}_nested_median_neighbour_560_{variant_label}_{threshold_label}_radiometry_available"] = False
        qc[f"raw_{quantity}_nested_median_neighbour_560_{variant_label}_relative_difference_{threshold_label}"] = np.nan
        qc[f"hard_filter_raw_{quantity}_nested_median_neighbour_560_{variant_label}_difference_gt_{threshold_label}"] = False

    spectrum_path = find_radiometry_spectrum(campaign_dir, campaign, quantity)
    if spectrum_path is None:
        return qc

    spectrum, sample_times = read_spectrum(spectrum_path)
    metadata_indices, deltas = nearest_metadata_index(sample_times, local_naive_times(df))
    matched = pd.DataFrame(
        {
            "sample_pos": np.arange(len(sample_times)),
            "metadata_index": metadata_indices,
            "time_delta_seconds": deltas,
        }
    )
    matched = matched[
        (matched["metadata_index"] >= 0)
        & (matched["time_delta_seconds"] <= match_tolerance_seconds)
    ].copy()
    if matched.empty:
        return qc

    matched["transect_nr"] = pd.to_numeric(df.loc[matched["metadata_index"], "transect_nr"].to_numpy(), errors="coerce")
    matched = matched[np.isfinite(matched["transect_nr"]) & (matched["transect_nr"] != -1)]
    if matched.empty:
        return qc

    for _, group in matched.groupby("transect_nr"):
        sample_positions = group["sample_pos"].to_numpy(dtype=int)
        differences_by_threshold, flags_by_threshold, resolved_wavelength = (
            nested_iterative_median_neighbour_relative_difference_at_wavelength(
                spectrum.iloc[:, sample_positions],
                sample_times[sample_positions],
                max_interval_seconds,
                threshold_fractions,
                target_wavelength_nm,
                max_wavelength_difference_nm,
            )
        )
        if resolved_wavelength is None:
            continue

        for threshold_label in threshold_fractions:
            matched_threshold = pd.DataFrame(
                {
                    "metadata_index": group["metadata_index"].to_numpy(dtype=int),
                    "difference": differences_by_threshold[threshold_label],
                    "sample_flagged": flags_by_threshold[threshold_label],
                }
            )
            by_metadata = matched_threshold.groupby("metadata_index").agg(
                difference=("difference", "max"),
                sample_flagged=("sample_flagged", "max"),
            )
            available = np.isfinite(by_metadata["difference"].to_numpy(dtype=float)) | by_metadata["sample_flagged"].astype(bool).to_numpy()
            qc.loc[by_metadata.index, f"raw_{quantity}_nested_median_neighbour_560_{variant_label}_{threshold_label}_radiometry_available"] = available
            qc.loc[by_metadata.index, f"raw_{quantity}_nested_median_neighbour_560_{variant_label}_relative_difference_{threshold_label}"] = by_metadata["difference"]
            qc.loc[by_metadata.index, f"hard_filter_raw_{quantity}_nested_median_neighbour_560_{variant_label}_difference_gt_{threshold_label}"] = by_metadata["sample_flagged"].astype(bool)

    return qc


def compute_quantity_neighbour_radiometry_qc(
    df: pd.DataFrame,
    campaign_dir: Path,
    campaign: str,
    quantity: str,
    threshold_fraction: float,
    match_tolerance_seconds: float,
    max_interval_seconds: float,
    rrs_wavelengths: np.ndarray,
) -> pd.DataFrame:
    qc = pd.DataFrame(index=df.index)
    qc[f"raw_{quantity}_neighbour_radiometry_available"] = False
    qc[f"raw_{quantity}_median_relative_neighbour_difference"] = np.nan
    qc[f"hard_filter_raw_{quantity}_neighbour_difference_gt_25pct"] = False

    spectrum_path = find_radiometry_spectrum(campaign_dir, campaign, quantity)
    if spectrum_path is None:
        return qc

    spectrum, sample_times = read_spectrum(spectrum_path)
    wl_min = float(np.nanmin(rrs_wavelengths))
    wl_max = float(np.nanmax(rrs_wavelengths))
    spectrum = spectrum.loc[(spectrum.index >= wl_min) & (spectrum.index <= wl_max)]
    if spectrum.empty:
        return qc

    differences, sample_flagged = iterative_neighbour_relative_difference(
        spectrum,
        sample_times,
        max_interval_seconds,
        threshold_fraction,
    )
    metadata_indices, deltas = nearest_metadata_index(sample_times, local_naive_times(df))
    matched = pd.DataFrame(
        {
            "metadata_index": metadata_indices,
            "time_delta_seconds": deltas,
            "difference": differences,
            "sample_flagged": sample_flagged,
        }
    )
    matched = matched[
        (matched["metadata_index"] >= 0)
        & (matched["time_delta_seconds"] <= match_tolerance_seconds)
        & np.isfinite(matched["difference"])
    ]
    if matched.empty:
        return qc

    by_metadata = matched.groupby("metadata_index").agg(
        difference=("difference", "max"),
        sample_flagged=("sample_flagged", "max"),
    )
    qc.loc[by_metadata.index, f"raw_{quantity}_neighbour_radiometry_available"] = True
    qc.loc[by_metadata.index, f"raw_{quantity}_median_relative_neighbour_difference"] = by_metadata["difference"]
    qc.loc[by_metadata.index, f"hard_filter_raw_{quantity}_neighbour_difference_gt_25pct"] = by_metadata["sample_flagged"].astype(bool)
    return qc


def compute_quantity_median_neighbour_560_radiometry_qc(
    df: pd.DataFrame,
    campaign_dir: Path,
    campaign: str,
    quantity: str,
    threshold_label: str,
    threshold_fraction: float,
    match_tolerance_seconds: float,
    max_interval_seconds: float,
    target_wavelength_nm: float,
    max_wavelength_difference_nm: float,
) -> pd.DataFrame:
    qc = pd.DataFrame(index=df.index)
    qc[f"raw_{quantity}_median_neighbour_560_{threshold_label}_radiometry_available"] = False
    qc[f"raw_{quantity}_median_neighbour_560_{threshold_label}_wavelength_nm"] = np.nan
    qc[f"raw_{quantity}_median_neighbour_560_relative_difference_{threshold_label}"] = np.nan
    qc[f"hard_filter_raw_{quantity}_median_neighbour_560_difference_gt_{threshold_label}"] = False

    spectrum_path = find_radiometry_spectrum(campaign_dir, campaign, quantity)
    if spectrum_path is None:
        return qc

    spectrum, sample_times = read_spectrum(spectrum_path)
    differences, sample_flagged, resolved_wavelength = iterative_median_neighbour_relative_difference_at_wavelength(
        spectrum,
        sample_times,
        max_interval_seconds,
        threshold_fraction,
        target_wavelength_nm,
        max_wavelength_difference_nm,
    )
    if resolved_wavelength is None:
        return qc

    metadata_indices, deltas = nearest_metadata_index(sample_times, local_naive_times(df))
    matched = pd.DataFrame(
        {
            "metadata_index": metadata_indices,
            "time_delta_seconds": deltas,
            "difference": differences,
            "sample_flagged": sample_flagged,
        }
    )
    matched = matched[
        (matched["metadata_index"] >= 0)
        & (matched["time_delta_seconds"] <= match_tolerance_seconds)
        & np.isfinite(matched["difference"])
    ]
    if matched.empty:
        return qc

    by_metadata = matched.groupby("metadata_index").agg(
        difference=("difference", "max"),
        sample_flagged=("sample_flagged", "max"),
    )
    qc.loc[by_metadata.index, f"raw_{quantity}_median_neighbour_560_{threshold_label}_radiometry_available"] = True
    qc.loc[by_metadata.index, f"raw_{quantity}_median_neighbour_560_{threshold_label}_wavelength_nm"] = resolved_wavelength
    qc.loc[by_metadata.index, f"raw_{quantity}_median_neighbour_560_relative_difference_{threshold_label}"] = by_metadata["difference"]
    qc.loc[by_metadata.index, f"hard_filter_raw_{quantity}_median_neighbour_560_difference_gt_{threshold_label}"] = by_metadata["sample_flagged"].astype(bool)
    return qc


def compute_quantity_560_finite_radiometry_qc(
    df: pd.DataFrame,
    campaign_dir: Path,
    campaign: str,
    quantity: str,
    match_tolerance_seconds: float,
    target_wavelength_nm: float,
    max_wavelength_difference_nm: float,
) -> pd.DataFrame:
    qc = pd.DataFrame(index=df.index)
    qc[f"raw_{quantity}_560_finite_available"] = False
    qc[f"raw_{quantity}_560_wavelength_nm"] = np.nan
    qc[f"raw_{quantity}_560_value"] = np.nan
    qc[f"hard_filter_raw_{quantity}_560_nonfinite"] = False

    spectrum_path = find_radiometry_spectrum(campaign_dir, campaign, quantity)
    if spectrum_path is None:
        return qc

    spectrum, sample_times = read_spectrum(spectrum_path)
    wavelengths = spectrum.index.to_numpy(dtype=float)
    target_idx = resolve_nearest_wavelength_index(
        wavelengths,
        target_wavelength_nm,
        max_wavelength_difference_nm,
    )
    if target_idx is None:
        qc[f"hard_filter_raw_{quantity}_560_nonfinite"] = True
        return qc

    values = spectrum.to_numpy(dtype=float)[target_idx, :]
    metadata_indices, deltas = nearest_metadata_index(sample_times, local_naive_times(df))
    matched = pd.DataFrame(
        {
            "metadata_index": metadata_indices,
            "time_delta_seconds": deltas,
            "value": values,
        }
    )
    matched = matched[
        (matched["metadata_index"] >= 0)
        & (matched["time_delta_seconds"] <= match_tolerance_seconds)
    ]
    if matched.empty:
        return qc

    by_metadata = matched.groupby("metadata_index").agg(value=("value", "first"))
    qc.loc[by_metadata.index, f"raw_{quantity}_560_finite_available"] = True
    qc.loc[by_metadata.index, f"raw_{quantity}_560_wavelength_nm"] = float(wavelengths[target_idx])
    qc.loc[by_metadata.index, f"raw_{quantity}_560_value"] = by_metadata["value"]
    qc.loc[by_metadata.index, f"hard_filter_raw_{quantity}_560_nonfinite"] = ~np.isfinite(by_metadata["value"].to_numpy(dtype=float))
    return qc


def compute_neighbour_radiometry_qc(
    df: pd.DataFrame,
    campaign_dir: Path,
    campaign: str,
    threshold_fraction: float,
    match_tolerance_seconds: float,
    max_interval_seconds: float,
    rrs_columns: list[tuple[str, float]],
) -> pd.DataFrame:
    rrs_wavelengths = np.asarray([wl for _, wl in rrs_columns], dtype=float)
    qc = pd.concat(
        [
            compute_quantity_neighbour_radiometry_qc(
                df,
                campaign_dir,
                campaign,
                quantity,
                threshold_fraction,
                match_tolerance_seconds,
                max_interval_seconds,
                rrs_wavelengths,
            )
            for quantity in ("Ed", "Ld", "Lu")
        ],
        axis=1,
    )
    qc["raw_neighbour_radiometry_available"] = (
        qc["raw_Ed_neighbour_radiometry_available"]
        | qc["raw_Ld_neighbour_radiometry_available"]
        | qc["raw_Lu_neighbour_radiometry_available"]
    )
    qc["raw_median_relative_neighbour_difference"] = qc[
        [
            "raw_Ed_median_relative_neighbour_difference",
            "raw_Ld_median_relative_neighbour_difference",
            "raw_Lu_median_relative_neighbour_difference",
        ]
    ].max(axis=1, skipna=True)
    qc["hard_filter_raw_radiometry_neighbour_difference_gt_25pct"] = (
        qc["hard_filter_raw_Ed_neighbour_difference_gt_25pct"]
        | qc["hard_filter_raw_Ld_neighbour_difference_gt_25pct"]
        | qc["hard_filter_raw_Lu_neighbour_difference_gt_25pct"]
    )
    return qc


def compute_median_neighbour_560_radiometry_qc(
    df: pd.DataFrame,
    campaign_dir: Path,
    campaign: str,
    threshold_fractions: dict[str, float],
    match_tolerance_seconds: float,
    max_interval_seconds: float,
    target_wavelength_nm: float,
    max_wavelength_difference_nm: float,
) -> pd.DataFrame:
    threshold_qc = []
    for threshold_label, threshold_fraction in threshold_fractions.items():
        threshold_qc.extend(
            compute_quantity_median_neighbour_560_radiometry_qc(
                df,
                campaign_dir,
                campaign,
                quantity,
                threshold_label,
                threshold_fraction,
                match_tolerance_seconds,
                max_interval_seconds,
                target_wavelength_nm,
                max_wavelength_difference_nm,
            )
            for quantity in ("Ed", "Ld")
        )
    qc = pd.concat(threshold_qc, axis=1)
    finite_qc = pd.concat(
        [
            compute_quantity_560_finite_radiometry_qc(
                df,
                campaign_dir,
                campaign,
                quantity,
                match_tolerance_seconds,
                target_wavelength_nm,
                max_wavelength_difference_nm,
            )
            for quantity in ("Ed", "Ld", "Lu")
        ],
        axis=1,
    )
    qc = pd.concat([qc, finite_qc], axis=1)
    qc["hard_filter_raw_radiometry_560_nonfinite"] = (
        qc["hard_filter_raw_Ed_560_nonfinite"]
        | qc["hard_filter_raw_Ld_560_nonfinite"]
        | qc["hard_filter_raw_Lu_560_nonfinite"]
    )
    for threshold_label in threshold_fractions:
        qc[f"raw_median_neighbour_560_{threshold_label}_radiometry_available"] = (
            qc[f"raw_Ed_median_neighbour_560_{threshold_label}_radiometry_available"]
            | qc[f"raw_Ld_median_neighbour_560_{threshold_label}_radiometry_available"]
        )
        qc[f"raw_median_neighbour_560_relative_difference_{threshold_label}"] = qc[
            [
                f"raw_Ed_median_neighbour_560_relative_difference_{threshold_label}",
                f"raw_Ld_median_neighbour_560_relative_difference_{threshold_label}",
            ]
        ].max(axis=1, skipna=True)
        qc[f"hard_filter_raw_radiometry_median_neighbour_560_difference_gt_{threshold_label}"] = (
            qc[f"hard_filter_raw_Ed_median_neighbour_560_difference_gt_{threshold_label}"]
            | qc[f"hard_filter_raw_Ld_median_neighbour_560_difference_gt_{threshold_label}"]
        )
        qc[f"hard_filter_raw_radiometry_median_neighbour_560_gt{threshold_label}_or_nonfinite"] = (
            qc[f"hard_filter_raw_radiometry_median_neighbour_560_difference_gt_{threshold_label}"]
            | qc["hard_filter_raw_radiometry_560_nonfinite"]
        )
    return qc


def compute_nested_median_neighbour_560_radiometry_qc(
    df: pd.DataFrame,
    campaign_dir: Path,
    campaign: str,
    variant_label: str,
    threshold_fractions: dict[str, float],
    match_tolerance_seconds: float,
    max_interval_seconds: float,
    target_wavelength_nm: float,
    max_wavelength_difference_nm: float,
) -> pd.DataFrame:
    qc = pd.concat(
        [
            compute_quantity_nested_median_neighbour_560_radiometry_qc(
                df,
                campaign_dir,
                campaign,
                quantity,
                variant_label,
                threshold_fractions,
                match_tolerance_seconds,
                max_interval_seconds,
                target_wavelength_nm,
                max_wavelength_difference_nm,
            )
            for quantity in ("Ed", "Ld")
        ],
        axis=1,
    )
    finite_qc = pd.concat(
        [
            compute_quantity_560_finite_radiometry_qc(
                df,
                campaign_dir,
                campaign,
                quantity,
                match_tolerance_seconds,
                target_wavelength_nm,
                max_wavelength_difference_nm,
            )
            for quantity in ("Ed", "Ld", "Lu")
        ],
        axis=1,
    )
    qc = pd.concat([qc, finite_qc], axis=1)
    qc["hard_filter_raw_radiometry_560_nonfinite"] = (
        qc["hard_filter_raw_Ed_560_nonfinite"]
        | qc["hard_filter_raw_Ld_560_nonfinite"]
        | qc["hard_filter_raw_Lu_560_nonfinite"]
    )

    # Combined availability (raw_Ed_..._available | raw_Ld_..._available) and the
    # combined max(Ed, Ld) relative difference are intentionally not materialised:
    # availability always matches both per-quantity columns exactly (it depends only
    # on time-matching, not on quantity), and the combined difference is diagnostic
    # only — the actual hard filter below is the OR of the per-quantity flags, not a
    # re-threshold of the max difference. Inspect the per-quantity columns instead.
    for threshold_label in threshold_fractions:
        qc[f"hard_filter_raw_radiometry_nested_median_neighbour_560_{variant_label}_difference_gt_{threshold_label}"] = (
            qc[f"hard_filter_raw_Ed_nested_median_neighbour_560_{variant_label}_difference_gt_{threshold_label}"]
            | qc[f"hard_filter_raw_Ld_nested_median_neighbour_560_{variant_label}_difference_gt_{threshold_label}"]
        )

    return qc


def discover_eligibility_files(insitu_root: Path, campaign: str | None = None) -> list[dict[str, object]]:
    files = []
    for campaign_dir in sorted(path for path in insitu_root.iterdir() if path.is_dir() and re.fullmatch(r"\d{8}_[A-Za-z0-9]+", path.name)):
        if campaign and campaign_dir.name != campaign:
            continue
        qc_dir = campaign_dir / "visualization_analysis" / "quality_control"
        if not qc_dir.exists():
            continue
        for path in sorted(qc_dir.glob(f"{campaign_dir.name}_metadata_with_Rrs_*_eligibility_qc.csv")):
            match = ELIGIBILITY_PATTERN.match(path.name)
            if match:
                files.append({"campaign": match.group("campaign"), "sensor": match.group("sensor").upper(), "path": path})
    return files


def output_path_for(input_csv: Path) -> Path:
    return input_csv.with_name(input_csv.name.replace("_eligibility_qc.csv", "_hard_filter_qc.csv"))


def drop_existing_qc_columns(df: pd.DataFrame) -> pd.DataFrame:
    generated_prefixes = (
        "hard_filter_",
        "temporal_",
        "metric_",
        "flag_",
        "bottom_reflectance_",
    )
    generated_exact = {
        "quality_flag",
        "quality_flag_label",
        "flags_evaluated",
        "flag_count_without_hard_filter",
        "flag_count_incl_hard_filter",
    }
    columns_to_drop = [
        col
        for col in df.columns
        if str(col).startswith(generated_prefixes) or str(col) in generated_exact
    ]
    return df.drop(columns=columns_to_drop, errors="ignore")


def process_file(
    input_csv: Path,
    campaign: str,
    sensor: str,
    delimiter: str | None,
    rrs_column_regex: str | None,
    wl_min: float,
    wl_max: float,
    required_wavelengths: tuple[float, ...],
    max_wavelength_difference_nm: float,
    eo_input_root: Path,
    radiometry_neighbour_threshold_fraction: float,
    radiometry_match_tolerance_seconds: float,
    radiometry_neighbour_max_interval_seconds: float,
) -> dict[str, object]:
    df, sep = read_table(input_csv, delimiter)
    source_df = drop_existing_qc_columns(df)
    campaign_dir = INSITU_ROOT / campaign

    rrs_columns = find_rrs_columns(source_df, wl_min, wl_max, rrs_column_regex)
    required_columns = resolve_required_wavelength_columns(rrs_columns, required_wavelengths, max_wavelength_difference_nm)

    eligibility_excluded = source_df["eligibility_exclude"].astype("boolean").fillna(True) if "eligibility_exclude" in source_df.columns else pd.Series(False, index=source_df.index)
    geometry_qc = compute_geometry_qc(source_df)
    spectral_qc = compute_spectral_qc(source_df, rrs_columns, required_columns)
    nested_median_neighbour_560_180s_qc = compute_nested_median_neighbour_560_radiometry_qc(
        source_df,
        campaign_dir,
        campaign,
        "180s",
        RADIOMETRY_NESTED_MEDIAN_NEIGHBOUR_THRESHOLDS,
        radiometry_match_tolerance_seconds,
        RADIOMETRY_MEDIAN_NEIGHBOUR_MAX_INTERVAL_SECONDS,
        RADIOMETRY_MEDIAN_NEIGHBOUR_TARGET_WAVELENGTH_NM,
        RADIOMETRY_MEDIAN_NEIGHBOUR_MAX_WAVELENGTH_DIFFERENCE_NM,
    )
    nested_median_neighbour_560_transect_qc = compute_nested_median_neighbour_560_radiometry_qc(
        source_df,
        campaign_dir,
        campaign,
        "transect",
        RADIOMETRY_NESTED_MEDIAN_NEIGHBOUR_THRESHOLDS,
        radiometry_match_tolerance_seconds,
        float("inf"),
        RADIOMETRY_MEDIAN_NEIGHBOUR_TARGET_WAVELENGTH_NM,
        RADIOMETRY_MEDIAN_NEIGHBOUR_MAX_WAVELENGTH_DIFFERENCE_NM,
    )
    # The Ed/Ld/Lu finite-value check at 560 nm does not depend on the neighbourhood
    # variant, so it is identical between the 180s and transect calls above. Drop the
    # transect copy to avoid duplicate columns (pandas would otherwise suffix them
    # with ".1" since both calls emit the same column names).
    duplicated_finite_qc_columns = {
        f"raw_{quantity}_560_{field}"
        for quantity in ("Ed", "Ld", "Lu")
        for field in ("finite_available", "wavelength_nm", "value")
    } | {f"hard_filter_raw_{quantity}_560_nonfinite" for quantity in ("Ed", "Ld", "Lu")} | {
        "hard_filter_raw_radiometry_560_nonfinite"
    }
    nested_median_neighbour_560_transect_qc = nested_median_neighbour_560_transect_qc.drop(
        columns=[
            col
            for col in nested_median_neighbour_560_transect_qc.columns
            if col in duplicated_finite_qc_columns
        ],
        errors="ignore",
    )
    temporal_qc = compute_temporal_qc(source_df, campaign, sensor, eo_input_root)
    result = pd.concat(
        [
            source_df.copy(),
            geometry_qc,
            spectral_qc,
            nested_median_neighbour_560_180s_qc,
            nested_median_neighbour_560_transect_qc,
            temporal_qc,
        ],
        axis=1,
    )
    base_reasons = pd.Series("", index=result.index)
    base_reasons = append_reason(base_reasons, eligibility_excluded, "eligibility_excluded")
    base_reasons = append_reason(base_reasons, result["hard_filter_vaz_outside_75_150"], "relative_azimuth_outside_75_150")
    base_reasons = append_reason(base_reasons, result["rrs_negative_baseline_shift"], "negative_baseline_shift")
    base_reasons = append_reason(base_reasons, result["rrs_positive_baseline_shift"], "positive_baseline_shift")
    base_reasons = append_reason(base_reasons, result["rrs_required_nonfinite"], "required_rrs_nonfinite")
    base_reasons = append_reason(base_reasons, result["rrs_required_nonpositive"], "required_rrs_nonpositive")
    # Original raw radiometry neighbour deviation filter retained as diagnostics only:
    # base_reasons = append_reason(base_reasons, result["hard_filter_raw_radiometry_neighbour_difference_gt_25pct"], "raw_radiometry_neighbour_difference_gt_25pct")
    base_reasons = append_reason(base_reasons, result["hard_filter_missing_satellite_overpass"], "missing_satellite_overpass")
    base_reasons = append_reason(base_reasons, result["hard_filter_temporal_offset_gt_180min"], "temporal_offset_gt_180min")

    for variant_label in ("180s", "transect"):
        result[f"hard_filter_raw_radiometry_nested_median_neighbour_560_{variant_label}_gt25pct_or_nonfinite"] = (
            result[f"hard_filter_raw_radiometry_nested_median_neighbour_560_{variant_label}_difference_gt_25pct"]
            | result["hard_filter_raw_radiometry_560_nonfinite"]
        )

        result[f"hard_filter_exclusion_reason_nested_{variant_label}"] = append_reason(
            base_reasons,
            result[f"hard_filter_raw_radiometry_nested_median_neighbour_560_{variant_label}_gt25pct_or_nonfinite"],
            f"raw_radiometry_nested_median_neighbour_560_{variant_label}_gt25pct_or_nonfinite",
        )
        result[f"hard_filter_exclude_nested_{variant_label}"] = result[f"hard_filter_exclusion_reason_nested_{variant_label}"].ne("")
        result[f"hard_filter_retain_nested_{variant_label}"] = ~result[f"hard_filter_exclude_nested_{variant_label}"]

    result["hard_filter_exclusion_reason"] = result["hard_filter_exclusion_reason_nested_180s"]
    result["hard_filter_exclude"] = result["hard_filter_exclude_nested_180s"]
    result["hard_filter_retain"] = result["hard_filter_retain_nested_180s"]
    result["hard_filter_exclusion_reason_strict"] = result["hard_filter_exclusion_reason_nested_transect"]
    result["hard_filter_exclude_strict"] = result["hard_filter_exclude_nested_transect"]
    result["hard_filter_retain_strict"] = result["hard_filter_retain_nested_transect"]

    # Summary counts are computed from the full result, before trimming columns
    # that aren't written to the output CSV (see OUTPUT_DROP_COLUMNS below).
    reason_counts = (
        result.loc[result["hard_filter_exclude"], "hard_filter_exclusion_reason"]
        .str.get_dummies(sep=";")
        .sum()
        .to_dict()
    )
    summary = {
        "campaign": campaign,
        "sensor": sensor,
        "rows": int(len(result)),
        "retained": int(result["hard_filter_retain"].sum()),
        "excluded": int(result["hard_filter_exclude"].sum()),
        "retained_strict": int(result["hard_filter_retain_strict"].sum()),
        "excluded_strict": int(result["hard_filter_exclude_strict"].sum()),
        "eligibility_excluded": int(eligibility_excluded.sum()),
        "vaz_outside_75_150": int(result["hard_filter_vaz_outside_75_150"].sum()),
        "negative_baseline_shift": int(result["rrs_negative_baseline_shift"].sum()),
        "positive_baseline_shift": int(result["rrs_positive_baseline_shift"].sum()),
        "required_rrs_nonfinite": int(result["rrs_required_nonfinite"].sum()),
        "required_rrs_nonpositive": int(result["rrs_required_nonpositive"].sum()),
        "raw_radiometry_560_nonfinite": int(result["hard_filter_raw_radiometry_560_nonfinite"].sum()),
        "raw_radiometry_nested_median_neighbour_560_180s_gt25pct_or_nonfinite": int(result["hard_filter_raw_radiometry_nested_median_neighbour_560_180s_gt25pct_or_nonfinite"].sum()),
        "raw_radiometry_nested_median_neighbour_560_transect_gt25pct_or_nonfinite": int(result["hard_filter_raw_radiometry_nested_median_neighbour_560_transect_gt25pct_or_nonfinite"].sum()),
        "temporal_offset_gt_180min": int(result["hard_filter_temporal_offset_gt_180min"].sum()),
        "required_rrs_columns": ";".join(f"{wl:g}:{col}" for wl, col in required_columns.items()),
        **{f"reason_{key}": int(value) for key, value in reason_counts.items()},
    }

    output_csv = output_path_for(input_csv)
    output_df = result.drop(columns=OUTPUT_DROP_COLUMNS, errors="ignore")
    output_df.to_csv(output_csv, sep=sep, index=False)
    summary["input_csv"] = str(input_csv)
    summary["output_csv"] = str(output_csv)
    return summary


def parse_required_wavelengths(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one wavelength is required.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply hard exclusion filters to eligibility QC tables.")
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--insitu-root", type=Path, default=INSITU_ROOT)
    parser.add_argument("--eo-input-root", type=Path, default=EO_INPUT_ROOT)
    parser.add_argument("--campaign", default=None)
    parser.add_argument("--delimiter", default=None)
    parser.add_argument("--rrs-column-regex", default=DEFAULT_RRS_COLUMN_REGEX)
    parser.add_argument("--wl-min", type=float, default=DEFAULT_WL_MIN)
    parser.add_argument("--wl-max", type=float, default=DEFAULT_WL_MAX)
    parser.add_argument("--required-wavelengths", type=parse_required_wavelengths, default=DEFAULT_REQUIRED_WAVELENGTHS)
    parser.add_argument("--max-wavelength-difference-nm", type=float, default=DEFAULT_MAX_WAVELENGTH_DIFFERENCE_NM)
    parser.add_argument("--radiometry-neighbour-threshold-fraction", type=float, default=RADIOMETRY_NEIGHBOUR_THRESHOLD_FRACTION)
    parser.add_argument("--radiometry-match-tolerance-seconds", type=float, default=RADIOMETRY_MATCH_TOLERANCE_SECONDS)
    parser.add_argument("--radiometry-neighbour-max-interval-seconds", type=float, default=RADIOMETRY_NEIGHBOUR_MAX_INTERVAL_SECONDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global INSITU_ROOT
    INSITU_ROOT = args.insitu_root

    if args.input_csv is not None:
        match = ELIGIBILITY_PATTERN.match(args.input_csv.name)
        if match is None:
            raise SystemExit(f"Input is not a canonical eligibility QC file: {args.input_csv}")
        summary = process_file(
            args.input_csv,
            match.group("campaign"),
            match.group("sensor").upper(),
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
        print(f"Processed {summary['input_csv']}: {summary['retained']} retained / {summary['rows']} rows -> {summary['output_csv']}")
        return

    files = discover_eligibility_files(args.insitu_root, args.campaign)
    if not files:
        suffix = f" for campaign {args.campaign}" if args.campaign else ""
        raise SystemExit(f"No eligibility QC files found below {args.insitu_root}{suffix}")

    summaries = []
    errors = []
    for meta in files:
        try:
            summary = process_file(
                Path(meta["path"]),
                str(meta["campaign"]),
                str(meta["sensor"]),
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
            summaries.append(summary)
            print(f"Processed {summary['input_csv']}: {summary['retained']} retained / {summary['rows']} rows")
        except Exception as exc:
            errors.append({"input_csv": str(meta["path"]), "error": str(exc)})
            print(f"Failed {meta['path']}: {exc}")

    print("Done.")
    print(f"Discovered eligibility files: {len(files)}")
    print(f"Successfully processed: {len(summaries)}")
    print(f"Failed: {len(errors)}")
    if errors:
        for error in errors:
            print(f"{error['input_csv']}: {error['error']}")


if __name__ == "__main__":
    main()
