"""
Bottom-reflectance risk screening for hard-filtered in situ Rrs spectra.

The QC decision is made on optical depth at the water's spectrally transparent
window - the waveband of lowest attenuation, where light penetrates deepest and
bottom reflectance therefore matters most (Cannizzaro & Carder 2006):

    lambda_tr = argmax of the campaign-median Rrs over 400-700 nm
    Kd_tr     = ln((0.14 - Rrs_tr) / 0.013) / (2.5 * Secchi)   Lee et al. 2015 Eq.33
    tau       = Kd_tr * depth_m

    tau >= 2.5   optically deep   (retained; ~0.193% two-way transmission is the boundary)
    tau < 2.5    shallow risk     (excluded)

Bottom-reflected radiance is attenuated down at Kd and back up at ~1.5*Kd for a
Lambertian bottom (Kirk 1991), so contribution scales with exp(-2.5 * tau). That
same 2.5 is the one in Eq. 33: Lee et al. treat the Secchi disk as a Lambertian
bottom, so the law governing whether a disk stays visible at depth z is the law
governing whether the lake bottom contaminates Rrs at depth z. Secchi depth is
thus a direct measurement of the relevant attenuation, not a loose proxy.

The threshold is that of the SeaWiFS protocols (Mueller & Austin 1995): water
deeper than 2.5 attenuation lengths as their general station rule (p.44). They
specify 490 nm, which approximates the open-ocean window; we apply the same
criterion at each campaign's own lambda_tr. Mueller & Austin also give a
stricter six-optical-depth rule aimed specifically at avoiding sea-bed
reflection (p.31); that would classify 2.5 <= tau < 6 as "transitional" rather
than deep. It is not applied here - only the 2.5 threshold is - so those
measurements are retained as optically deep.

Kd_tr replaces the common Kd = 1.7/Secchi (Poole & Atkins 1929), which returns
KPAR - a broadband coefficient dominated by strongly attenuating wavelengths. In
these campaigns KPAR exceeds Kd_tr by 1.80-1.86x, which would overestimate the
optical depth relevant to bottom contamination.

A spectral-shape metric is also computed and written, but only as a **diagnostic**
- it never drives exclusion. It follows Cannizzaro & Carder (2006), adapted to
inland waters and to Sentinel-2 MSI / Landsat OLI compatible bands:

    x = log(Rrs443 / Rrs665)                blue:red ratio, a colour axis
    y = log(Rrs443 * Rrs665 / Rrs560^2)     curvature at 560 nm

A campaign-specific LOWESS curve fitted through the optically deep spectra gives
the expected curvature at each colour; the residual is standardised by its median
absolute deviation and sign-reversed, so a high `bottom_spectral_z` means an
anomalously strong green peak - the direction bottom reflectance would push it.

Why it is diagnostic only: measured against the optical-depth labels, the spectral
score reached 0.566 balanced accuracy on 20240618_BIE, the one campaign with
enough optically shallow rows (130) to test it. On the rest the threshold would be
fitted on 0-4 positive examples. Separately, of the 426 measurements lacking
bathymetry across all campaigns, 424 are already excluded by other QC criteria, so
a spectral fallback would change the outcome for a single measurement.

Processing:
  1. Read *_hard_filter_qc.csv from each campaign quality-control folder.
  2. Classify by tau where depth is available.
  3. Compute the spectral diagnostic where the three bands are usable.
  4. Write a compact QC table plus per-campaign diagnostic plots.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


INSITU_ROOT = Path(r"C:\Users\puehrifi\Documents\insitu")
SCRIPT_DIR = Path(__file__).resolve().parent
BATCH_SUMMARY_CSV = SCRIPT_DIR / "bottom_reflectance_batch_summary.csv"

# Authoritative Secchi depths. Edit this file, not the dict below: it is used
# automatically whenever it is present, so a correction here reaches the QC run
# without needing --secchi-table.
DEFAULT_SECCHI_TABLE = SCRIPT_DIR / "bottom_reflectance_secchi_depths.csv"

# Fallback only, for when the table above is missing. Keep in sync with it.
DEFAULT_SECCHI_DEPTH_M = {
    "20230610_CST": 2.4,
    "20231008_ZRH": 4.2,
    "20240618_BIE": 5.0,
    "20240619_WAL": 4.0,
    "20250303_CST": 10.0,
    "20260226_ZRH": float(np.mean([7.6, 7.7])),
    "20260227_ZRH": 7.5,
    "20260423_ZRH": float(np.mean([5.4, 5.3])),
    "20260430_ZRH": 6.0,
}

HARD_FILTER_PATTERN = re.compile(
    r"^(?P<campaign>\d{8}_[A-Za-z0-9]+)_metadata_with_Rrs_(?P<sensor>L8|L9|S2A|S2B)(?:_[A-Za-z0-9]+)*_hard_filter_qc\.csv$",
    re.IGNORECASE,
)

TARGET_WAVELENGTHS = (443.0, 560.0, 665.0)
MAX_WAVELENGTH_DIFFERENCE_NM = 2.0
LOWESS_FRAC = 0.25
MIN_REFERENCE_POINTS = 30
MAD_SCALE = 1.4826
ROBUST_REFERENCE_ITERATIONS = 2
ROBUST_REFERENCE_NEGATIVE_Z_CUTOFF = -2.5

# Mueller & Austin (1995): 2.5 attenuation lengths is the general station rule
# (p.44). Their stricter 6-optical-depth rule against sea-bed reflection
# (p.31) is not applied - tau >= OPTICAL_SHALLOW_TAU is classified as deep.
OPTICAL_SHALLOW_TAU = 2.5

# Lee et al. (2015) Eq. 33 constants.
#   DISK_TERM = (R_T / pi) * (t^2 / n^2) = (0.85 / pi) * 0.54, white Secchi disk
#   EYE_CONTRAST_THRESHOLD = C_r^t, the eye's detection threshold
#   BOTTOM_PATH_FACTOR = Kd down + 1.5*Kd up for a Lambertian bottom (Kirk 1991)
DISK_TERM = 0.14
EYE_CONTRAST_THRESHOLD = 0.013
BOTTOM_PATH_FACTOR = 2.5

# Transparent window is located on the campaign-median spectrum over this range;
# a per-spectrum extremum would be biased low by noise.
WINDOW_MIN_NM, WINDOW_MAX_NM = 400.0, 700.0

# Columns carried into the hard-filter table. Everything else the stage computes
# is either a per-file constant (see the batch summary), an exact algebraic
# duplicate, or recomputable from the Rrs columns already present.
BOTTOM_COLUMNS = ["bottom_tau", "bottom_class", "bottom_spectral_z", "bottom_exclude"]
# Identifiers kept so the written table can be joined and inspected on its own.
ID_COLUMNS = [
    "fid_ramses",
    "timestamp (UTC)",
    "datetime (UTC)",
    "datetime (CET)",
    "transect_nr",
    "depth_m",
]

BATCH_SUMMARY_COLUMNS = [
    "campaign",
    "sensor",
    "input_csv",
    "output_csv",
    "rows",
    "secchi_depth_m",
    "secchi_source",
    "lambda_tr_nm",
    "rrs_tr",
    "kd_tr_m-1",
    "kd_tr_x_secchi",
    "window_rows",
    "depth_available",
    "deep",
    "shallow_risk",
    "unknown",
    "invalid_spectral",
    "excluded",
    "retained",
    "spectral_valid_rows",
    "reference_rows",
    "reference_source",
    "residual_median",
    "residual_scale",
    "residual_scale_source",
]


@dataclass
class ProcessedFile:
    meta: dict[str, object]
    df: pd.DataFrame
    sep: str
    output_dir: Path
    summary_base: dict[str, object]
    x_fit: np.ndarray
    y_fit: np.ndarray


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
    df.columns = [str(col).strip() for col in df.columns]
    return df, sep


def to_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(series, errors="coerce")


def parse_rrs_wavelength(column_name: object) -> float | None:
    match = re.fullmatch(r"Rrs_([0-9]+(?:\.[0-9]+)?)", str(column_name).strip())
    return float(match.group(1)) if match else None


def resolve_rrs_columns(df: pd.DataFrame) -> dict[float, str]:
    available: list[tuple[str, float]] = []
    for col in df.columns:
        wavelength = parse_rrs_wavelength(col)
        if wavelength is not None:
            available.append((str(col), wavelength))

    if not available:
        raise ValueError("No Rrs_<wavelength> columns found.")

    resolved: dict[float, str] = {}
    for target in TARGET_WAVELENGTHS:
        selected_col, selected_wl = min(available, key=lambda item: abs(item[1] - target))
        diff = abs(selected_wl - target)
        if diff > MAX_WAVELENGTH_DIFFERENCE_NM:
            raise ValueError(
                f"No hyperspectral Rrs column within {MAX_WAVELENGTH_DIFFERENCE_NM:g} nm of {target:g} nm. "
                f"Nearest is {selected_col} ({selected_wl:g} nm), {diff:g} nm away."
            )
        resolved[target] = selected_col
    return resolved


def discover_hard_filter_qc(root: Path, campaign_filter: str | None = None) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for campaign_dir in sorted(root.iterdir()):
        if not campaign_dir.is_dir() or not re.match(r"^\d{8}_", campaign_dir.name):
            continue
        if campaign_filter and campaign_dir.name != campaign_filter:
            continue

        qc_dir = campaign_dir / "visualization_analysis" / "quality_control"
        if not qc_dir.exists():
            continue

        for csv_path in sorted(qc_dir.glob("*_hard_filter_qc.csv")):
            if any(token in csv_path.name for token in ("_conv_", "_current", "_example", "_old")):
                continue
            match = HARD_FILTER_PATTERN.match(csv_path.name)
            if not match:
                continue
            files.append(
                {
                    "campaign": match.group("campaign"),
                    "sensor": match.group("sensor").upper(),
                    "path": csv_path,
                    "campaign_dir": campaign_dir,
                }
            )
    return files


def load_secchi_table(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Secchi table not found: {path}")
    df, _ = read_table(path)
    if "campaign" not in df.columns or "secchi_depth_m" not in df.columns:
        raise ValueError(f"Secchi table must contain campaign and secchi_depth_m columns: {path}")

    df["secchi_depth_m"] = to_numeric(df["secchi_depth_m"])
    df = df[df["secchi_depth_m"].notna() & (df["secchi_depth_m"] > 0)]
    # Mean, matching the "Mean ZSD" convention reported in the methods. The
    # shipped table is already one pre-aggregated row per campaign, so this only
    # matters if raw per-cast readings are added later.
    return df.groupby("campaign")["secchi_depth_m"].mean().to_dict()


def resolve_secchi_by_campaign(secchi_table: Path | None) -> tuple[dict[str, float], str]:
    """Secchi depth per campaign, preferring the on-disk table over the fallback."""
    if secchi_table is not None:
        return load_secchi_table(secchi_table), str(secchi_table)
    if DEFAULT_SECCHI_TABLE.exists():
        return load_secchi_table(DEFAULT_SECCHI_TABLE), str(DEFAULT_SECCHI_TABLE)
    return dict(DEFAULT_SECCHI_DEPTH_M), "built_in_default"


def unique_mean_curve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    curve = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    curve = curve.groupby("x", as_index=False)["y"].mean().sort_values("x")
    return curve["x"].to_numpy(), curve["y"].to_numpy()


def lowess_curve(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    import statsmodels.api as sm

    reference = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(reference) < 3:
        raise ValueError("Fewer than three valid spectra available for LOWESS reference fitting.")

    curve = sm.nonparametric.lowess(
        reference["y"],
        reference["x"],
        frac=LOWESS_FRAC,
        return_sorted=True,
    )
    x_fit, y_fit = unique_mean_curve(curve[:, 0], curve[:, 1])
    if len(x_fit) < 2:
        x0 = float(reference["x"].iloc[0])
        y0 = float(np.nanmedian(reference["y"]))
        eps = max(abs(x0) * 1e-6, 1e-6)
        return np.asarray([x0 - eps, x0 + eps]), np.asarray([y0, y0])
    return x_fit, y_fit


def expected_from_curve(x: pd.Series, x_fit: np.ndarray, y_fit: np.ndarray) -> pd.Series:
    out = pd.Series(np.nan, index=x.index, dtype=float)
    valid = x.notna() & np.isfinite(x)
    if valid.any():
        clipped = x.loc[valid].clip(float(np.min(x_fit)), float(np.max(x_fit)))
        out.loc[valid] = np.interp(clipped, x_fit, y_fit)
    return out


def robust_location_scale(values: pd.Series) -> tuple[float, float, str]:
    finite = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return np.nan, np.nan, "none"

    median = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - median))) * MAD_SCALE
    if np.isfinite(mad) and mad > 0:
        return median, mad, "mad"

    std = float(np.nanstd(finite))
    if np.isfinite(std) and std > 0:
        return median, std, "std_fallback"

    return median, 1.0, "unit_fallback"


def rrs_spectrum_columns(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Rrs columns within the search window, ascending in wavelength."""
    pairs = []
    for col in df.columns:
        wavelength = parse_rrs_wavelength(col)
        if wavelength is not None and WINDOW_MIN_NM <= wavelength <= WINDOW_MAX_NM:
            pairs.append((wavelength, str(col)))
    pairs.sort()
    return np.array([w for w, _ in pairs], dtype=float), [c for _, c in pairs]


def transparent_window(df: pd.DataFrame, usable: pd.Series) -> tuple[float, float]:
    """(lambda_tr, Rrs_tr) from the campaign-median spectrum.

    Taking the median across measurements first, then the maximum across
    wavelengths, keeps noise from biasing the window: a per-spectrum argmax would
    systematically land on whichever band happened to be noisy high.
    """
    wavelengths, columns = rrs_spectrum_columns(df)
    if len(wavelengths) < 3 or not usable.any():
        return float("nan"), float("nan")

    spectra = df.loc[usable, columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        median_spectrum = np.nanmedian(spectra, axis=0)
    if not np.isfinite(median_spectrum).any():
        return float("nan"), float("nan")

    peak = int(np.nanargmax(median_spectrum))
    return float(wavelengths[peak]), float(median_spectrum[peak])


def kd_transparent_window(secchi_depth_m: float | None, rrs_tr: float) -> float:
    """Kd at the transparent window, from Lee et al. (2015) Eq. 33 inverted.

    Z_SD = ln((0.14 - Rrs_tr) / 0.013) / (2.5 * Kd_tr)  ->  solve for Kd_tr.
    Reduces to roughly 0.93/Z_SD for natural waters, because Rrs_tr is small
    next to the disk term and the logarithm barely moves.
    """
    if secchi_depth_m is None or secchi_depth_m <= 0 or not np.isfinite(rrs_tr):
        return float("nan")
    contrast = DISK_TERM - rrs_tr
    if contrast <= EYE_CONTRAST_THRESHOLD:
        # Water brighter than the disk contrast: the model has no solution.
        return float("nan")
    return float(
        np.log(contrast / EYE_CONTRAST_THRESHOLD) / (BOTTOM_PATH_FACTOR * secchi_depth_m)
    )


def optical_depth(df: pd.DataFrame, kd_tr: float) -> pd.Series:
    """tau = Kd_tr * depth, NaN where Kd_tr or bathymetric depth is missing."""
    if not np.isfinite(kd_tr) or "depth_m" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)

    tau = kd_tr * to_numeric(df["depth_m"])
    return tau.where(np.isfinite(tau))


def fit_reference(df: pd.DataFrame, valid: pd.Series, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray, pd.Series, str]:
    """Reference curve for the diagnostic, preferring optically deep spectra."""
    optical_deep = valid & df["bottom_tau"].ge(OPTICAL_SHALLOW_TAU)
    if int(optical_deep.sum()) >= MIN_REFERENCE_POINTS:
        x_fit, y_fit = lowess_curve(df.loc[optical_deep, x_col], df.loc[optical_deep, y_col])
        return x_fit, y_fit, optical_deep, f"optical_depth_deep_tau_ge_{OPTICAL_SHALLOW_TAU:g}"

    reference = valid.copy()
    reference_source = "robust_iterative_all_valid"
    for _ in range(ROBUST_REFERENCE_ITERATIONS):
        if int(reference.sum()) < MIN_REFERENCE_POINTS:
            reference = valid.copy()
            reference_source = "all_valid_min_reference_fallback"
            break

        x_fit, y_fit = lowess_curve(df.loc[reference, x_col], df.loc[reference, y_col])
        residual = df[y_col] - expected_from_curve(df[x_col], x_fit, y_fit)
        median, scale, _ = robust_location_scale(residual.loc[reference])
        updated = valid & (((residual - median) / scale) >= ROBUST_REFERENCE_NEGATIVE_Z_CUTOFF)
        if int(updated.sum()) < MIN_REFERENCE_POINTS:
            break
        reference = updated

    x_fit, y_fit = lowess_curve(df.loc[reference, x_col], df.loc[reference, y_col])
    return x_fit, y_fit, reference, reference_source


def compute_bottom_qc(meta: dict[str, object], secchi_depth_m: float | None, delimiter: str | None) -> ProcessedFile:
    csv_path = Path(meta["path"])
    df, sep = read_table(csv_path, delimiter)
    df = df.copy()

    rrs_cols = resolve_rrs_columns(df)
    rrs_443, rrs_560, rrs_665 = (rrs_cols[w] for w in TARGET_WAVELENGTHS)
    for col in (rrs_443, rrs_560, rrs_665):
        df[col] = to_numeric(df[col])

    spectral_ok = (
        df[[rrs_443, rrs_560, rrs_665]].notna().all(axis=1)
        & (df[rrs_443] > 0)
        & (df[rrs_560] > 0)
        & (df[rrs_665] > 0)
    )

    # ---- QC criterion: optical depth at the transparent window ----
    # Locate the window on measurements that passed the preceding hard filters
    # where possible, so a handful of bad spectra cannot drag the median.
    window_rows = spectral_ok
    if "hard_filter_retain" in df.columns:
        retained = df["hard_filter_retain"].astype(str).str.strip().str.lower() == "true"
        if int((spectral_ok & retained).sum()) >= MIN_REFERENCE_POINTS:
            window_rows = spectral_ok & retained

    lambda_tr, rrs_tr = transparent_window(df, window_rows)
    kd_tr = kd_transparent_window(secchi_depth_m, rrs_tr)
    depth = to_numeric(df["depth_m"]) if "depth_m" in df.columns else pd.Series(np.nan, index=df.index, dtype=float)
    missing_depth = depth.isna()

    df["bottom_tau"] = optical_depth(df, kd_tr)
    tau = df["bottom_tau"]
    has_tau = tau.notna()

    x_feature = pd.Series(np.nan, index=df.index, dtype=float)
    y_feature = pd.Series(np.nan, index=df.index, dtype=float)
    x_feature.loc[spectral_ok] = np.log(df.loc[spectral_ok, rrs_443] / df.loc[spectral_ok, rrs_665])
    y_feature.loc[spectral_ok] = np.log(
        df.loc[spectral_ok, rrs_443] * df.loc[spectral_ok, rrs_665] / df.loc[spectral_ok, rrs_560] ** 2
    )
    df["_x"] = x_feature
    df["_y"] = y_feature
    finite_valid = spectral_ok & np.isfinite(x_feature) & np.isfinite(y_feature)

    df["bottom_spectral_z"] = np.nan
    median = scale = np.nan
    scale_source = "none"
    x_fit = y_fit = np.asarray([])
    reference_mask = pd.Series(False, index=df.index)
    reference_source = "insufficient_valid_spectra"

    if int(finite_valid.sum()) >= 3:
        x_fit, y_fit, reference_mask, reference_source = fit_reference(df, finite_valid, "_x", "_y")
        residual = df["_y"] - expected_from_curve(df["_x"], x_fit, y_fit)
        median, scale, scale_source = robust_location_scale(residual.loc[reference_mask])
        # sign-reversed: high score = weaker curvature than expected = stronger
        # green peak, the direction bottom reflectance pushes the spectrum
        df.loc[finite_valid, "bottom_spectral_z"] = -((residual.loc[finite_valid] - median) / scale)

    # ---- single class, depth first ----
    df["bottom_class"] = "unknown"
    df.loc[has_tau & tau.ge(OPTICAL_SHALLOW_TAU), "bottom_class"] = "deep"
    df.loc[has_tau & tau.lt(OPTICAL_SHALLOW_TAU), "bottom_class"] = "shallow_risk"
    # Only where depth cannot decide does spectral usability matter. Missing
    # bathymetry is still excluded below because bottom-reflectance risk cannot
    # be cleared without depth information.
    df.loc[~has_tau & ~finite_valid, "bottom_class"] = "invalid_spectral"

    df["bottom_exclude"] = df["bottom_class"].isin(["shallow_risk", "invalid_spectral"]) | missing_depth

    output_dir = Path(meta["campaign_dir"]) / "visualization_analysis" / "quality_control" / "bottom_reflectance"
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = df["bottom_class"].value_counts()
    summary_base = {
        "campaign": meta["campaign"],
        "sensor": meta["sensor"],
        "input_csv": str(csv_path),
        "rows": int(len(df)),
        "secchi_depth_m": secchi_depth_m if secchi_depth_m is not None else np.nan,
        "lambda_tr_nm": lambda_tr,
        "rrs_tr": rrs_tr,
        "kd_tr_m-1": kd_tr,
        # Kd_tr * Z_SD; ~0.93 for natural waters, a quick sanity check on Eq. 33.
        "kd_tr_x_secchi": kd_tr * secchi_depth_m if secchi_depth_m and np.isfinite(kd_tr) else np.nan,
        "window_rows": int(window_rows.sum()),
        "depth_available": int((~missing_depth).sum()),
        "deep": int(counts.get("deep", 0)),
        "shallow_risk": int(counts.get("shallow_risk", 0)),
        "unknown": int(counts.get("unknown", 0)),
        "invalid_spectral": int(counts.get("invalid_spectral", 0)),
        "excluded": int(df["bottom_exclude"].sum()),
        "retained": int((~df["bottom_exclude"]).sum()),
        "spectral_valid_rows": int(finite_valid.sum()),
        "reference_rows": int(reference_mask.sum()),
        "reference_source": reference_source,
        "residual_median": median,
        "residual_scale": scale,
        "residual_scale_source": scale_source,
    }
    return ProcessedFile(meta, df, sep, output_dir, summary_base, x_fit, y_fit)


def percentile_limits(values: pd.Series, lower: float = 2, upper: float = 98) -> tuple[float, float] | None:
    finite = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return None

    lo, hi = np.nanpercentile(finite, [lower, upper])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return None

    pad = (hi - lo) * 0.05
    return float(lo - pad), float(hi + pad)


CLASS_COLORS = {
    "deep": "tab:blue",
    "shallow_risk": "tab:red",
    "unknown": "tab:green",
    "invalid_spectral": "0.6",
}


def plot_classified_space(item: ProcessedFile, campaign: str, file_stem: str) -> None:
    import matplotlib.pyplot as plt

    df = item.df
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls, group in df.groupby("bottom_class"):
        ax.scatter(group["_x"], group["_y"], s=10, alpha=0.55,
                   label=f"{cls} n={len(group)}", color=CLASS_COLORS.get(cls, "black"))

    xlim = percentile_limits(df["_x"])
    ylim = percentile_limits(df["_y"])
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    if len(item.x_fit):
        x_min, x_max = ax.get_xlim()
        keep = (item.x_fit >= x_min) & (item.x_fit <= x_max)
        if keep.any():
            ax.plot(item.x_fit[keep], item.y_fit[keep], color="black", lw=2,
                    label="LOWESS reference", zorder=5)
    ax.set_xlabel("log(Rrs443 / Rrs665)")
    ax.set_ylabel("log((Rrs443 * Rrs665) / Rrs560^2)")
    ax.set_title(f"{campaign}: bottom classes (from optical depth)")
    ax.legend(fontsize=8)
    plt.savefig(item.output_dir / f"{file_stem}_bottom_reflectance_classified_space.png",
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_tau_vs_spectral(item: ProcessedFile, campaign: str, file_stem: str) -> None:
    """Diagnostic: does the spectral score track optical depth at all?"""
    import matplotlib.pyplot as plt

    df = item.df
    usable = df["bottom_tau"].notna() & df["bottom_spectral_z"].notna()
    if not usable.any():
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df.loc[usable, "bottom_tau"], df.loc[usable, "bottom_spectral_z"],
               s=10, alpha=0.5, color="tab:blue")
    ax.axvline(OPTICAL_SHALLOW_TAU, color="tab:red", lw=1.5, label=f"tau={OPTICAL_SHALLOW_TAU:g} (shallow risk)")
    ax.set_xscale("log")
    ylim = percentile_limits(df.loc[usable, "bottom_spectral_z"])
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel("optical depth tau = Kd_tr * z")
    ax.set_ylabel("bottom_spectral_z (diagnostic)")
    ax.set_title(f"{campaign}: spectral diagnostic vs optical depth")
    ax.legend(fontsize=8)
    plt.savefig(item.output_dir / f"{file_stem}_bottom_reflectance_tau_vs_spectral.png",
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_transect_histograms(item: ProcessedFile, campaign: str, file_stem: str) -> None:
    import matplotlib.pyplot as plt

    df = item.df
    if "transect_nr" not in df.columns:
        return

    hist_df = df.copy()
    hist_df["_t"] = to_numeric(hist_df["transect_nr"])
    hist_df = hist_df[hist_df["_t"].notna() & (hist_df["_t"] != -1)]
    hist_df = hist_df.dropna(subset=["bottom_tau"])
    if hist_df.empty:
        return

    transects = sorted(hist_df["_t"].unique())
    ncols = 2
    nrows = int(np.ceil(len(transects) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, max(4, 3 * nrows)), squeeze=False)

    for ax, transect in zip(axes.ravel(), transects):
        values = hist_df.loc[hist_df["_t"] == transect, "bottom_tau"]
        ax.hist(values, bins=30, color="tab:blue", alpha=0.75)
        ax.axvline(OPTICAL_SHALLOW_TAU, color="tab:red", lw=1.5)
        ax.set_title(f"transect {int(transect)}")
        ax.set_xlabel("tau")
        ax.set_ylabel("n")

    for ax in axes.ravel()[len(transects):]:
        ax.axis("off")

    fig.suptitle(f"{campaign}: optical depth by transect")
    plt.tight_layout()
    plt.savefig(item.output_dir / f"{file_stem}_bottom_reflectance_tau_by_transect.png",
                dpi=300, bbox_inches="tight")
    plt.close()


def write_outputs(item: ProcessedFile) -> dict[str, object]:
    csv_path = Path(item.meta["path"])
    plot_stem = csv_path.stem.replace("_hard_filter_qc", "").replace("_hard_filter_retained", "")
    output_csv = item.output_dir / f"{plot_stem}_bottom_reflectance_qc.csv"

    columns = [c for c in ID_COLUMNS if c in item.df.columns] + BOTTOM_COLUMNS
    item.df[columns].to_csv(output_csv, sep=item.sep, index=False)

    plot_classified_space(item, str(item.meta["campaign"]), plot_stem)
    plot_tau_vs_spectral(item, str(item.meta["campaign"]), plot_stem)
    plot_transect_histograms(item, str(item.meta["campaign"]), plot_stem)

    summary = dict(item.summary_base)
    summary["output_csv"] = str(output_csv)
    return summary


def write_summary(summaries: list[dict[str, object]], campaign_filter: str | None) -> None:
    new_summary = pd.DataFrame(summaries).reindex(columns=BATCH_SUMMARY_COLUMNS)
    if campaign_filter and BATCH_SUMMARY_CSV.exists():
        old_summary, _ = read_table(BATCH_SUMMARY_CSV)
        if "campaign" in old_summary.columns:
            old_summary = old_summary[old_summary["campaign"] != campaign_filter]
            # Older runs used a different schema; keep only shared columns so the
            # file stays readable rather than silently gaining empty ones.
            old_summary = old_summary.reindex(columns=BATCH_SUMMARY_COLUMNS)
            new_summary = pd.concat([old_summary, new_summary], ignore_index=True)
    new_summary.to_csv(BATCH_SUMMARY_CSV, sep=";", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bottom-reflectance QC from optical depth, with a spectral diagnostic."
    )
    parser.add_argument("--insitu-root", type=Path, default=INSITU_ROOT)
    parser.add_argument("--campaign", help="Optional campaign filter, for example 20260430_ZRH.")
    parser.add_argument("--delimiter", default=None, help="Optional CSV delimiter. Auto-detected by default.")
    parser.add_argument(
        "--secchi-table",
        type=Path,
        default=None,
        help=(
            "Optional CSV with campaign and secchi_depth_m columns. "
            "If omitted, built-in campaign Secchi depths are used."
        ),
    )
    parser.add_argument("--secchi-summary", type=Path, dest="secchi_table", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    secchi_by_campaign, secchi_source = resolve_secchi_by_campaign(args.secchi_table)
    files = discover_hard_filter_qc(args.insitu_root, args.campaign)

    if not files:
        suffix = f" for campaign {args.campaign}" if args.campaign else ""
        raise SystemExit(f"No hard-filter QC hyperspectral files found below {args.insitu_root}{suffix}")

    summaries: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for meta in files:
        try:
            item = compute_bottom_qc(meta, secchi_by_campaign.get(str(meta["campaign"])), args.delimiter)
            item.summary_base["secchi_source"] = secchi_source
            summaries.append(write_outputs(item))
            base = item.summary_base
            print(
                f"{meta['campaign']} {meta['sensor']}: "
                f"lambda_tr={base['lambda_tr_nm']:.0f}nm Kd_tr={base['kd_tr_m-1']:.4f} "
                f"(Kd_tr*ZSD={base['kd_tr_x_secchi']:.3f}) | deep={base['deep']} "
                f"shallow_risk={base['shallow_risk']} "
                f"unknown={base['unknown']} invalid={base['invalid_spectral']} -> "
                f"excluded={base['excluded']}"
            )
        except Exception as exc:
            errors.append({"input_csv": str(meta["path"]), "error": str(exc)})
            print(f"Failed: {meta['path']} -> {exc}")

    write_summary(summaries, args.campaign)
    print(f"\nProcessed {len(summaries)} files, {len(errors)} failed.")
    print(f"Batch summary: {BATCH_SUMMARY_CSV}")
    for error in errors:
        print(f"{error['input_csv']}: {error['error']}")


if __name__ == "__main__":
    main()
