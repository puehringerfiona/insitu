from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .processing import wavelength_columns


QC_GOOD = 1
QC_SPIKE_OR_SHAPE = 3
QC_NEGATIVE = 4
QC_NAN = 9


@dataclass(frozen=True)
class ACSQCConfig:
    max_wavelength_nm: float = 720.0
    rolling_window: int = 11
    spike_mad_multiplier: float = 6.0
    min_valid_fraction: float = 0.65
    absorption_shape_log_ratio: float = 0.35
    green_band_start_nm: float = 565.0
    green_band_end_nm: float = 610.0
    green_band_left_count: int = 4
    green_band_right_count: int = 5


@dataclass(frozen=True)
class ACSQCResult:
    data: pd.DataFrame
    flags: pd.DataFrame
    summary: pd.DataFrame


def apply_acs_qc(df: pd.DataFrame, config: ACSQCConfig | None = None) -> ACSQCResult:
    """Clean ACS absorption/attenuation columns in a synchronized dataframe.

    The returned data keeps the original non-ACS columns. ACS values are replaced
    with NaN when flagged, interpolated along the profile/time axis, and then
    lightly smoothed with a centered rolling median. Absorption also gets a
    green-band offset correction modelled after the older acsa_complete_qc_v02.
    """
    config = config or ACSQCConfig()
    cleaned = df.copy()
    flag_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    a_cols = wavelength_columns(df, "a_")
    c_cols = wavelength_columns(df, "c_")

    if a_cols:
        a_clean, a_flags, a_summary = _clean_acs_block(
            df[a_cols],
            prefix="a_",
            config=config,
            use_absorption_shape_qc=True,
            apply_green_band_shift=True,
        )
        cleaned.loc[:, a_cols] = a_clean
        flag_frames.append(a_flags)
        summaries.append(a_summary)

    if c_cols:
        c_clean, c_flags, c_summary = _clean_acs_block(
            df[c_cols],
            prefix="c_",
            config=config,
            use_absorption_shape_qc=False,
            apply_green_band_shift=False,
        )
        cleaned.loc[:, c_cols] = c_clean
        flag_frames.append(c_flags)
        summaries.append(c_summary)

    flags = pd.concat(flag_frames, axis=1) if flag_frames else pd.DataFrame(index=df.index)
    flags.index.name = df.index.name
    summary = pd.DataFrame(summaries)
    return ACSQCResult(data=cleaned, flags=flags, summary=summary)


def _clean_acs_block(
    block: pd.DataFrame,
    prefix: str,
    config: ACSQCConfig,
    use_absorption_shape_qc: bool,
    apply_green_band_shift: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    wavelengths = np.array([float(col.removeprefix(prefix)) for col in block.columns], dtype=float)
    values = block.apply(pd.to_numeric, errors="coerce").astype(float)
    flags = pd.DataFrame(QC_GOOD, index=values.index, columns=values.columns, dtype="int16")

    original_nan = values.isna()
    flags = flags.mask(original_nan, QC_NAN)

    outside_range = wavelengths > config.max_wavelength_nm
    if outside_range.any():
        outside_cols = list(values.columns[outside_range])
        values.loc[:, outside_cols] = np.nan
        flags.loc[:, outside_cols] = QC_SPIKE_OR_SHAPE

    negative = values < 0
    values = values.mask(negative)
    flags = flags.mask(negative, QC_NEGATIVE)

    spike_mask = _rolling_spike_mask(values, config.rolling_window, config.spike_mad_multiplier)
    values = values.mask(spike_mask)
    flags = flags.mask(spike_mask, QC_SPIKE_OR_SHAPE)

    if use_absorption_shape_qc:
        shape_mask = _absorption_shape_mask(values, wavelengths, config)
        values = values.mask(shape_mask)
        flags = flags.mask(shape_mask & (flags == QC_GOOD), QC_SPIKE_OR_SHAPE)

    values = _interpolate_profile_gaps(values)

    if apply_green_band_shift:
        values = _correct_green_band_shift(values, wavelengths, config)

    values = _smooth_profile(values, config.rolling_window)

    valid_fraction = values.notna().mean(axis=1)
    bad_rows = valid_fraction < config.min_valid_fraction
    if bad_rows.any():
        values.loc[bad_rows, :] = np.nan
        flags.loc[bad_rows, :] = QC_SPIKE_OR_SHAPE

    summary = {
        "channel": "absorption" if prefix == "a_" else "attenuation",
        "columns": len(block.columns),
        "rows": len(block),
        "max_wavelength_nm": config.max_wavelength_nm,
        "nan_flags": int((flags == QC_NAN).sum().sum()),
        "negative_flags": int((flags == QC_NEGATIVE).sum().sum()),
        "spike_or_shape_flags": int((flags == QC_SPIKE_OR_SHAPE).sum().sum()),
        "finite_values_after_qc": int(np.isfinite(values.to_numpy(dtype=float)).sum()),
    }
    return values, flags.add_prefix("qc_"), summary


def _rolling_spike_mask(values: pd.DataFrame, window: int, multiplier: float) -> pd.DataFrame:
    window = _safe_odd_window(window, len(values))
    if window < 3 or values.empty:
        return pd.DataFrame(False, index=values.index, columns=values.columns)

    center = values.rolling(window=window, center=True, min_periods=max(3, window // 2)).median()
    residual = values - center
    mad = residual.abs().rolling(window=window, center=True, min_periods=max(3, window // 2)).median()
    scale = 1.4826 * mad

    fallback = residual.abs().median(axis=0, skipna=True).replace(0, np.nan)
    scale = scale.fillna(fallback)
    threshold = multiplier * scale
    return (residual.abs() > threshold).fillna(False)


def _absorption_shape_mask(values: pd.DataFrame, wavelengths: np.ndarray, config: ACSQCConfig) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=values.index, columns=values.columns)
    fit_cols = (wavelengths >= 400) & (wavelengths <= 600)
    check_cols = wavelengths <= 430
    if fit_cols.sum() < 4 or check_cols.sum() == 0:
        return mask

    fit_wl = wavelengths[fit_cols]
    check_wl = wavelengths[check_cols]
    fit_col_names = values.columns[fit_cols]
    check_col_names = values.columns[check_cols]

    for idx, row in values.iterrows():
        y = row.loc[fit_col_names].to_numpy(dtype=float)
        positive = np.isfinite(y) & (y > 0)
        if positive.sum() < 4:
            mask.loc[idx, check_col_names] = True
            continue

        x_fit = fit_wl[positive]
        log_y = np.log(y[positive])
        slope, intercept = np.polyfit(x_fit - 400.0, log_y, deg=1)
        predicted = np.exp(intercept + slope * (check_wl - 400.0))
        observed = row.loc[check_col_names].to_numpy(dtype=float)
        valid = np.isfinite(observed) & (observed > 0)
        row_bad = np.zeros(len(check_col_names), dtype=bool)
        row_bad[valid] = np.abs(np.log(predicted[valid] / observed[valid])) > config.absorption_shape_log_ratio
        row_bad[~valid] = True
        mask.loc[idx, check_col_names] = row_bad

    return mask


def _interpolate_profile_gaps(values: pd.DataFrame) -> pd.DataFrame:
    out = values.copy()
    out = out.interpolate(axis=0, method="index", limit_direction="both")
    out = out.interpolate(axis=1, limit_direction="both")
    return out


def _correct_green_band_shift(values: pd.DataFrame, wavelengths: np.ndarray, config: ACSQCConfig) -> pd.DataFrame:
    cols = values.columns
    band = (wavelengths >= config.green_band_start_nm) & (wavelengths <= config.green_band_end_nm)
    if band.sum() < config.green_band_left_count + config.green_band_right_count + 2:
        return values

    band_positions = np.where(band)[0]
    left_positions = band_positions[: config.green_band_left_count]
    right_positions = band_positions[-config.green_band_right_count :]
    adjust_positions = np.where(wavelengths < config.green_band_start_nm)[0]
    band_col_names = cols[band_positions]
    out = values.copy()

    for idx, row in out.iterrows():
        right_y = row.iloc[right_positions].to_numpy(dtype=float)
        right_x = wavelengths[right_positions]
        left_y = row.iloc[left_positions].to_numpy(dtype=float)
        if np.isfinite(right_y).sum() < 2 or np.isfinite(left_y).sum() < 2:
            continue
        finite_right = np.isfinite(right_y)
        slope, intercept = np.polyfit(right_x[finite_right], right_y[finite_right], deg=1)
        expected_left = slope * np.nanmean(wavelengths[left_positions]) + intercept
        shift = expected_left - np.nanmean(left_y)
        if not np.isfinite(shift):
            continue

        out.loc[idx, cols[adjust_positions]] = row.iloc[adjust_positions] + shift
        band_y = row.loc[band_col_names].to_numpy(dtype=float)
        shifted_band_y = band_y.copy()
        shifted_band_y[: config.green_band_left_count] = shifted_band_y[: config.green_band_left_count] + shift
        valid = np.isfinite(shifted_band_y)
        if valid.sum() >= 2:
            out.loc[idx, band_col_names] = np.interp(wavelengths[band_positions], wavelengths[band_positions][valid], shifted_band_y[valid])

    return out


def _smooth_profile(values: pd.DataFrame, window: int) -> pd.DataFrame:
    window = _safe_odd_window(window, len(values))
    if window < 3:
        return values
    return values.rolling(window=window, center=True, min_periods=max(3, window // 2)).median().interpolate(
        axis=0,
        method="index",
        limit_direction="both",
    )


def _safe_odd_window(window: int, length: int) -> int:
    if length <= 0:
        return 0
    window = max(1, min(int(window), int(length)))
    if window % 2 == 0:
        window -= 1
    return max(1, window)
