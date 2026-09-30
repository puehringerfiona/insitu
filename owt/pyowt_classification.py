"""
Classify in situ Rrs spectra with the Bi and Hieronymi pyOWT scheme.

This wrapper keeps the CSV-oriented workflow used in this project, but delegates
the optical-variable calculation and Mahalanobis membership classification to
the vendored pyOWT implementation in pyOWT.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PYOWT_ROOT = SCRIPT_DIR / "pyOWT"
if str(PYOWT_ROOT) not in sys.path:
    sys.path.insert(0, str(PYOWT_ROOT))

from pyowt.OWT import OWT  # noqa: E402
from pyowt.OpticalVariables import OpticalVariables  # noqa: E402


TYPE_DESCRIPTIONS = {
    "1": "Extremely clear and oligotrophic indigo-blue waters.",
    "2": "Blue waters with slightly higher detritus and CDOM than OWT 1.",
    "3a": "Turquoise waters with moderate phytoplankton, detritus, and CDOM.",
    "3b": "Bright 3a-like waters with strong scattering, e.g. coccolithophore blooms.",
    "4a": "Greenish coastal or inland waters with higher biomass.",
    "4b": "Bright green 4a-like bloom waters with higher scattering.",
    "5a": "Green eutrophic waters with typical peaks near 560 and 709 nm.",
    "5b": "Green hyper-eutrophic waters with a near-infrared reflectance plateau.",
    "6": "Bright brown waters dominated by high detrital scattering.",
    "7": "Dark brown to black waters dominated by very high CDOM absorption.",
    "NaN": "Not classifiable at the configured membership threshold.",
}

SENSOR_ALIASES = {
    "L8": "OLI-landsat-8",
    "L9": "OLI-landsat-8",
    "S2A": "msi-sentinel-2a",
    "S2B": "msi-sentinel-2b",
    "S3A": "olci-s3a",
    "S3B": "olci-s3b",
}


@dataclass(frozen=True)
class SpectralTable:
    metadata: pd.DataFrame
    spectra: np.ndarray
    wavelengths: np.ndarray
    wavelength_columns: list[str]


def read_csv_with_delimiter(path: Path, delimiter: str | None = None) -> tuple[pd.DataFrame, str]:
    if delimiter is not None:
        return pd.read_csv(path, sep=delimiter), delimiter

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        sample = file.read(16384)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        detected = dialect.delimiter
        df = pd.read_csv(path, sep=detected)
        if len(df.columns) > 1:
            return df, detected
    except csv.Error:
        pass

    for candidate in [",", ";", "\t", "|"]:
        df = pd.read_csv(path, sep=candidate)
        if len(df.columns) > 1:
            return df, candidate

    raise ValueError("Could not detect CSV delimiter. Pass --delimiter explicitly.")


def parse_wavelength_from_column(column: object, prefix_regex: str | None = None) -> float | None:
    text = str(column).strip()
    if prefix_regex and re.search(prefix_regex, text, flags=re.IGNORECASE) is None:
        return None
    matches = re.findall(r"(?<!\d)(\d{3}(?:\.\d+)?)(?!\d)", text)
    if not matches:
        return None
    return float(matches[-1])


def parse_true_like(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "t", "yes", "y", "1"})


def apply_quality_filter(df: pd.DataFrame, column: str | None, minimum: float | None) -> pd.DataFrame:
    if not column:
        return df
    if column not in df.columns:
        raise ValueError(f"Quality filter column not found: {column}")
    text = df[column].astype(str).str.strip().str.lower()
    boolean_tokens = {"true", "false", "t", "f", "yes", "no", "y", "n", "1", "0"}
    if minimum is None and text.dropna().isin(boolean_tokens).all():
        return df.loc[parse_true_like(df[column])].reset_index(drop=True)

    values = pd.to_numeric(df[column], errors="coerce")
    if minimum is None:
        return df.loc[values.notna()].reset_index(drop=True)
    if values.notna().any():
        return df.loc[values > minimum].reset_index(drop=True)
    return df.loc[parse_true_like(df[column])].reset_index(drop=True)


def extract_spectral_table(
    df: pd.DataFrame,
    wl_min: float,
    wl_max: float,
    rrs_column_regex: str | None,
    id_columns: Iterable[str] | None,
    reject_negative_rrs: bool,
) -> SpectralTable:
    candidates: list[tuple[str, float]] = []
    for col in df.columns:
        wl = parse_wavelength_from_column(col, rrs_column_regex)
        if wl is not None and wl_min <= wl <= wl_max:
            candidates.append((str(col), wl))

    if not candidates:
        raise ValueError("No Rrs wavelength columns found in the selected wavelength range.")

    candidates = sorted(candidates, key=lambda item: item[1])
    wl_cols = [col for col, _ in candidates]
    wavelengths = np.asarray([wl for _, wl in candidates], dtype=float)
    spectra = df[wl_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(spectra).all(axis=1)
    has_any_finite = np.isfinite(spectra).any(axis=1)
    positive_peak = np.zeros(spectra.shape[0], dtype=bool)
    positive_peak[has_any_finite] = np.nanmax(spectra[has_any_finite, :], axis=1) > 0
    valid &= positive_peak
    if reject_negative_rrs:
        valid &= (spectra >= 0).all(axis=1)

    if id_columns:
        keep_metadata_cols = [c for c in id_columns if c in df.columns]
    else:
        keep_metadata_cols = [c for c in df.columns if c not in wl_cols]

    metadata = df.loc[valid, keep_metadata_cols].reset_index(drop=True).copy()
    spectra = spectra[valid, :]
    if len(spectra) == 0:
        raise ValueError("No valid spectra remain after removing missing, invalid, or all-zero rows.")

    return SpectralTable(metadata, spectra, wavelengths, wl_cols)


def trapezoid(y: np.ndarray, x: np.ndarray, axis: int) -> np.ndarray:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y=y, x=x, axis=axis)
    return np.trapz(y=y, x=x, axis=axis)


def area_normalize(spectra: np.ndarray, wavelengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    area = trapezoid(spectra, wavelengths, axis=1)
    normalized = np.full_like(spectra, np.nan, dtype=float)
    valid = np.isfinite(area) & (area > 0)
    normalized[valid, :] = spectra[valid, :] / area[valid, None]
    return normalized, area


def resolve_sensor(sensor: str | None) -> str | None:
    if sensor is None:
        return None
    return SENSOR_ALIASES.get(sensor.upper(), sensor)


def classify_pyowt(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    sensor: str | None,
    version: str,
    threshold: float,
) -> tuple[OpticalVariables, OWT]:
    try:
        ov = OpticalVariables(Rrs=spectra, band=wavelengths, sensor=sensor, version=version)
    except ValueError as exc:
        if sensor is None and (wavelengths.min() > 400 or wavelengths.max() < 800):
            raise ValueError(
                "pyOWT hyperspectral mode requires wavelengths spanning 400-800 nm. "
                "Use --wl-min 400 --wl-max 800 with a hyperspectral file, or pass --sensor "
                "for a supported multispectral band set."
            ) from exc
        raise
    owt = OWT(AVW=ov.AVW, Area=ov.Area, NDI=ov.NDI, version=version, thres_u=threshold)
    return ov, owt


def plot_type_counts(path: Path, counts: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(counts["owt_label"].astype(str), counts["n"], color="#4C78A8")
    max_n = counts["n"].max()
    for bar, n in zip(bars, counts["n"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(max_n * 0.015, 0.5),
            f"n={int(n)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.ylim(0, max_n * 1.14 if max_n > 0 else 1)
    plt.xlabel("pyOWT class")
    plt.ylabel("Spectra")
    plt.title("pyOWT class counts")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_mean_normalized_spectra(path: Path, wavelengths: np.ndarray, normalized: np.ndarray, labels: np.ndarray) -> None:
    plt.figure(figsize=(9, 5))
    for label in sorted(pd.unique(labels.astype(str))):
        mask = labels.astype(str) == label
        mean = np.nanmean(normalized[mask, :], axis=0)
        plt.plot(wavelengths, mean, label=label)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Area-normalized Rrs [nm^-1]")
    plt.title("Mean normalized spectrum per pyOWT class")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify Rrs spectra with the Bi and Hieronymi pyOWT scheme.")
    parser.add_argument("--input-csv", required=True, type=Path, help="Input CSV containing metadata and Rrs columns.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for outputs.")
    parser.add_argument("--delimiter", default=None, help="CSV delimiter. Omit for auto-detection.")
    parser.add_argument("--wl-min", type=float, default=400)
    parser.add_argument("--wl-max", type=float, default=800)
    parser.add_argument("--rrs-column-regex", default=r"^(?:Rrs_)?\d{3}(?:\.\d+)?(?:nm)?$")
    parser.add_argument("--id-columns", nargs="*", default=None, help="Metadata columns to keep. Default keeps all non-spectral columns.")
    parser.add_argument("--quality-column", default=None)
    parser.add_argument("--quality-min", type=float, default=None)
    parser.add_argument("--reject-negative-rrs", action="store_true")
    parser.add_argument("--sensor", default=None, help="pyOWT sensor name or alias: L8, L9, S2A, S2B, S3A, S3B.")
    parser.add_argument("--owt-version", choices=["v01", "v02"], default="v01")
    parser.add_argument("--membership-threshold", type=float, default=0.0001)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, delimiter = read_csv_with_delimiter(args.input_csv, args.delimiter)
    df = apply_quality_filter(df, args.quality_column, args.quality_min)
    table = extract_spectral_table(
        df=df,
        wl_min=args.wl_min,
        wl_max=args.wl_max,
        rrs_column_regex=args.rrs_column_regex,
        id_columns=args.id_columns,
        reject_negative_rrs=args.reject_negative_rrs,
    )

    sensor = resolve_sensor(args.sensor)
    ov, owt = classify_pyowt(
        spectra=table.spectra,
        wavelengths=table.wavelengths,
        sensor=sensor,
        version=args.owt_version,
        threshold=args.membership_threshold,
    )
    normalized, spectral_area = area_normalize(table.spectra, table.wavelengths)

    type_idx = owt.type_idx.reshape(-1).astype(int)
    labels = owt.type_str.reshape(-1).astype(str)
    classifiable = owt.classifiability.reshape(-1).astype(int)
    membership = owt.u.reshape(len(labels), -1)
    membership_best = np.full(len(labels), np.nan, dtype=float)
    finite_membership = np.isfinite(membership).any(axis=1)
    membership_best[finite_membership] = np.nanmax(membership[finite_membership, :], axis=1)

    classification = pd.DataFrame(
        {
            "owt_scheme": "pyOWT Bi and Hieronymi 2024",
            "owt_version": args.owt_version,
            "owt_label": labels,
            "owt_index": type_idx,
            "owt_classifiable": classifiable,
            "owt_membership_total": owt.utot.reshape(-1),
            "owt_membership_best": membership_best,
            "AVW": np.asarray(ov.AVW).reshape(-1),
            "Area": np.asarray(ov.Area).reshape(-1),
            "NDI": np.asarray(ov.NDI).reshape(-1),
            "spectral_area_raw": spectral_area,
        }
    )
    membership_df = pd.DataFrame(
        {f"membership_{label}": membership[:, idx] for idx, label in enumerate(owt.classInfo.typeName)}
    )
    normalized_df = pd.DataFrame(
        {f"Rrs_norm_{wl:g}": normalized[:, idx] for idx, wl in enumerate(table.wavelengths)}
    )
    result = pd.concat([table.metadata, classification, membership_df, normalized_df], axis=1)

    result.to_csv(args.output_dir / "owt_classified_spectra.csv", index=False)

    counts = result["owt_label"].value_counts(dropna=False).rename_axis("owt_label").reset_index(name="n")
    counts["fraction"] = counts["n"] / counts["n"].sum()
    counts["description"] = counts["owt_label"].astype(str).map(TYPE_DESCRIPTIONS).fillna("")
    counts.to_csv(args.output_dir / "owt_counts.csv", index=False)

    run_info = pd.DataFrame(
        [
            {"setting": "input_csv", "value": str(args.input_csv)},
            {"setting": "delimiter", "value": repr(delimiter)},
            {"setting": "classifier", "value": "pyOWT"},
            {"setting": "pyowt_root", "value": str(PYOWT_ROOT)},
            {"setting": "owt_version", "value": args.owt_version},
            {"setting": "sensor", "value": sensor or "hyperspectral"},
            {"setting": "membership_threshold", "value": args.membership_threshold},
            {"setting": "valid_spectra", "value": len(result)},
            {"setting": "classifiable_spectra", "value": int(result["owt_classifiable"].sum())},
            {"setting": "wavelength_min", "value": table.wavelengths.min()},
            {"setting": "wavelength_max", "value": table.wavelengths.max()},
            {"setting": "n_wavelengths", "value": len(table.wavelengths)},
            {"setting": "reference", "value": "Bi and Hieronymi (2024), doi:10.1002/lno.12606"},
        ]
    )
    run_info.to_csv(args.output_dir / "owt_run_info.csv", index=False)

    if not args.no_plots:
        plot_type_counts(args.output_dir / "01_owt_counts.png", counts)
        plot_mean_normalized_spectra(args.output_dir / "02_owt_mean_normalized_spectra.png", table.wavelengths, normalized, labels)

    print(f"Read {len(df)} rows from {args.input_csv} with delimiter {delimiter!r}.")
    print(f"Classified {len(result)} valid spectra with pyOWT {args.owt_version}.")
    print(f"Classifiable spectra: {int(result['owt_classifiable'].sum())}/{len(result)}.")
    print(f"Outputs written to {args.output_dir.resolve()}.")


if __name__ == "__main__":
    main()
