from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd


# Default campaign settings. Override these with CLI arguments when needed.
DEFAULT_META_PATH = Path(
    r"C:\Users\puehrifi\Documents\insitu\20260226_ZRH\final_metadata\20260226_ZRH_metadata_with_Rrs_L8_current_new.csv"
)
DEFAULT_RRS_PATH = Path(
    r"C:\Users\puehrifi\Documents\insitu\20260226_ZRH\final_metadata\current_like_bounds\Rrs_output_3C_O25.csv"
)
DEFAULT_SRF_PATH = Path(r"C:\Users\puehrifi\Documents\eo_data\SRF_files\OLI_SRF.csv")
DEFAULT_OUTPUT_PATH = Path(
    r"C:\Users\puehrifi\Documents\insitu\20260226_ZRH\final_metadata\20260226_ZRH_metadata_with_Rrs_L8_current_new.csv"
)
DEFAULT_SENSOR = "L8"  # options: L8, L9, S2A, S2B
DEFAULT_USE_ALL_WAVELENGTHS = True
DEFAULT_MERGE_TOLERANCE = "10s"
DEFAULT_TIMEZONE = "CET"


# === Band maps per sensor ===
L8_BAND_MAP = {
    "L8_B1": 443,
    "L8_B2": 483,
    "L8_B3": 561,
    "L8_B4": 655,
    "L8_B5": 865,
}

L9_BAND_MAP = {
    "L9_B1": 443,
    "L9_B2": 482,
    "L9_B3": 561,
    "L9_B4": 654,
    "L9_B5": 865,
}

S2A_BAND_MAP = {
    "S2A_SR_AV_B1": 443,
    "S2A_SR_AV_B2": 492,
    "S2A_SR_AV_B3": 560,
    "S2A_SR_AV_B4": 665,
    "S2A_SR_AV_B5": 704,
    "S2A_SR_AV_B6": 740,
    "S2A_SR_AV_B7": 783,
    "S2A_SR_AV_B8": 842,
    "S2A_SR_AV_B8A": 865,
}

S2B_BAND_MAP = {
    "S2B_SR_AV_B1": 442,
    "S2B_SR_AV_B2": 492,
    "S2B_SR_AV_B3": 559,
    "S2B_SR_AV_B4": 665,
    "S2B_SR_AV_B5": 704,
    "S2B_SR_AV_B6": 739,
    "S2B_SR_AV_B7": 780,
    "S2B_SR_AV_B8": 842,
    "S2B_SR_AV_B8A": 865,
}

BAND_MAPS = {
    "L8": L8_BAND_MAP,
    "L9": L9_BAND_MAP,
    "S2A": S2A_BAND_MAP,
    "S2B": S2B_BAND_MAP,
}

# Allow L9 to use L8-labelled OLI SRF files and S2B to use S2A-labelled MSI
# SRF files when separate SRF columns are not available.
SRF_COLUMN_MAPS = {
    "L8": {band: band for band in L8_BAND_MAP},
    "L9": {
        "L9_B1": "L8_B1",
        "L9_B2": "L8_B2",
        "L9_B3": "L8_B3",
        "L9_B4": "L8_B4",
        "L9_B5": "L8_B5",
    },
    "S2A": {band: band for band in S2A_BAND_MAP},
    "S2B": {
        "S2B_SR_AV_B1": "S2A_SR_AV_B1",
        "S2B_SR_AV_B2": "S2A_SR_AV_B2",
        "S2B_SR_AV_B3": "S2A_SR_AV_B3",
        "S2B_SR_AV_B4": "S2A_SR_AV_B4",
        "S2B_SR_AV_B5": "S2A_SR_AV_B5",
        "S2B_SR_AV_B6": "S2A_SR_AV_B6",
        "S2B_SR_AV_B7": "S2A_SR_AV_B7",
        "S2B_SR_AV_B8": "S2A_SR_AV_B8",
        "S2B_SR_AV_B8A": "S2A_SR_AV_B8A",
    },
}


def parse_args():
    parser = ArgumentParser(
        description=(
            "Append hyperspectral or SRF-convolved Rrs values to synchronized "
            "metadata by nearest timestamp."
        )
    )
    parser.add_argument("--meta-path", type=Path, default=DEFAULT_META_PATH)
    parser.add_argument("--rrs-path", type=Path, default=DEFAULT_RRS_PATH)
    parser.add_argument("--srf-path", type=Path, default=DEFAULT_SRF_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--sensor", choices=sorted(BAND_MAPS), default=DEFAULT_SENSOR)
    parser.add_argument("--merge-tolerance", default=DEFAULT_MERGE_TOLERANCE)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument(
        "--all-wavelengths",
        dest="use_all_wavelengths",
        action="store_true",
        default=DEFAULT_USE_ALL_WAVELENGTHS,
        help="Append all hyperspectral Rrs wavelengths.",
    )
    parser.add_argument(
        "--convolve",
        dest="use_all_wavelengths",
        action="store_false",
        help="Append SRF-convolved sensor bands instead of all wavelengths.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing output to the same path as the input metadata file.",
    )
    return parser.parse_args()


def ensure_timezone(obj, tz_name="CET"):
    """
    Parse datetimes robustly:
    - if naive -> localize
    - if aware -> convert
    Works for both pandas Series and DatetimeIndex.
    """
    dt = pd.to_datetime(obj, errors="coerce")

    if isinstance(dt, pd.Series):
        if dt.dt.tz is None:
            return dt.dt.tz_localize(tz_name)
        return dt.dt.tz_convert(tz_name)

    if isinstance(dt, pd.DatetimeIndex):
        if dt.tz is None:
            return dt.tz_localize(tz_name)
        return dt.tz_convert(tz_name)

    raise TypeError(f"Unsupported type: {type(obj)}")


def load_metadata(meta_path, timezone):
    df_meta = pd.read_csv(meta_path, sep=None, engine="python")
    df_meta.columns = df_meta.columns.str.strip().str.replace("\ufeff", "", regex=False)

    if "datetime (CET)" not in df_meta.columns:
        raise ValueError("Metadata file must contain a 'datetime (CET)' column.")

    df_meta["datetime (CET)"] = ensure_timezone(df_meta["datetime (CET)"], timezone)
    df_meta["datetime (CET)"] = pd.DatetimeIndex(df_meta["datetime (CET)"]).astype(
        f"datetime64[ns, {timezone}]"
    )
    return df_meta


def load_rrs(rrs_path, timezone):
    df_rrs = pd.read_csv(rrs_path, sep=None, engine="python", index_col=0)
    df_rrs.index = df_rrs.index.astype(float)
    df_rrs.columns = ensure_timezone(df_rrs.columns, timezone)

    valid_cols = pd.notna(df_rrs.columns)
    df_rrs = df_rrs.loc[:, valid_cols]
    df_rrs = df_rrs.loc[:, ~df_rrs.columns.duplicated()]
    df_rrs = df_rrs.sort_index()

    df_rrs.columns = ensure_timezone(df_rrs.columns, timezone)
    df_rrs.columns = pd.DatetimeIndex(df_rrs.columns).astype(f"datetime64[ns, {timezone}]")
    return df_rrs


def build_hyperspectral_table(df_rrs):
    df_simulated = pd.DataFrame({"datetime (CET)": df_rrs.columns})

    for wavelength in df_rrs.index.tolist():
        df_simulated[f"Rrs_{wavelength:g}"] = df_rrs.loc[wavelength].values

    return df_simulated


def build_convolved_table(df_rrs, srf_path, sensor):
    band_map = BAND_MAPS[sensor]
    selected_srf_columns = SRF_COLUMN_MAPS[sensor]
    df_srf = pd.read_csv(srf_path)

    if "SR_WL" not in df_srf.columns:
        raise ValueError("SRF file must contain a column named 'SR_WL'.")

    srf_wavelengths = df_srf["SR_WL"].astype(float).values
    measured_wavelengths = df_rrs.index.values.astype(float)

    available_band_cols = [
        band for band, srf_col in selected_srf_columns.items() if srf_col in df_srf.columns
    ]
    missing_band_cols = [
        band for band, srf_col in selected_srf_columns.items() if srf_col not in df_srf.columns
    ]

    if missing_band_cols:
        print(f"Warning: these requested bands were not found in the SRF file and will be skipped: {missing_band_cols}")

    if not available_band_cols:
        raise ValueError("None of the requested band columns were found in the SRF file.")

    interpolated_srfs = {}
    for band_name in available_band_cols:
        srf_col = selected_srf_columns[band_name]
        raw_srf = df_srf[srf_col].astype(float).values

        srf_interp = np.interp(
            measured_wavelengths,
            srf_wavelengths,
            raw_srf,
            left=0.0,
            right=0.0,
        )

        srf_area = np.trapezoid(srf_interp, measured_wavelengths)
        if srf_area <= 0:
            print(f"Warning: SRF area is zero for band {band_name}; skipping.")
            continue

        interpolated_srfs[band_name] = srf_interp

    if not interpolated_srfs:
        raise ValueError("No valid SRFs remained after interpolation.")

    simulated_dict = {"datetime (CET)": df_rrs.columns}

    for band_name, srf_interp in interpolated_srfs.items():
        band_values = []

        for timestamp in df_rrs.columns:
            rrs_spectrum = df_rrs[timestamp].astype(float).values

            valid = np.isfinite(rrs_spectrum) & np.isfinite(srf_interp)
            if valid.sum() < 2:
                band_values.append(np.nan)
                continue

            numerator = np.trapezoid(
                rrs_spectrum[valid] * srf_interp[valid],
                measured_wavelengths[valid],
            )
            denominator = np.trapezoid(srf_interp[valid], measured_wavelengths[valid])

            if denominator <= 0:
                band_values.append(np.nan)
            else:
                band_values.append(numerator / denominator)

        wavelength_center = band_map[band_name]
        simulated_dict[f"Rrs_{wavelength_center:g}"] = band_values

    return pd.DataFrame(simulated_dict)


def merge_rrs_to_metadata(df_meta, df_simulated, merge_tolerance):
    df_simulated = df_simulated.sort_values("datetime (CET)")
    df_meta = df_meta.sort_values("datetime (CET)").reset_index(drop=True)
    new_rrs_cols = [col for col in df_simulated.columns if col != "datetime (CET)"]

    for col in new_rrs_cols:
        if col in df_meta.columns:
            df_meta = df_meta.drop(columns=[col])

        df_band = df_simulated[["datetime (CET)", col]]
        df_meta = pd.merge_asof(
            df_meta,
            df_band,
            on="datetime (CET)",
            direction="nearest",
            tolerance=pd.Timedelta(merge_tolerance),
        )

    return df_meta, new_rrs_cols


def main():
    args = parse_args()

    if args.meta_path.resolve() == args.output_path.resolve() and not args.overwrite:
        raise SystemExit(
            "Output path is identical to metadata input path. Use a different "
            "--output-path or pass --overwrite intentionally."
        )

    df_meta = load_metadata(args.meta_path, args.timezone)
    df_rrs = load_rrs(args.rrs_path, args.timezone)

    if args.use_all_wavelengths:
        df_simulated = build_hyperspectral_table(df_rrs)
    else:
        df_simulated = build_convolved_table(df_rrs, args.srf_path, args.sensor)

    df_merged, new_rrs_cols = merge_rrs_to_metadata(
        df_meta,
        df_simulated,
        args.merge_tolerance,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(args.output_path, index=False)

    mode = "hyperspectral wavelengths" if args.use_all_wavelengths else f"{args.sensor} convolved bands"
    print(f"Added {len(new_rrs_cols)} Rrs columns ({mode}).")
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
