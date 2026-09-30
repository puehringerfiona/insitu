from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .io import add_prefix


@dataclass(frozen=True)
class ProfileSegments:
    all: pd.DataFrame
    down: pd.DataFrame
    up: pd.DataFrame


def interpolate_to_ctd_time(ctd: pd.DataFrame, sensor: pd.DataFrame) -> pd.DataFrame:
    """Interpolate a sensor dataframe onto the CTD timestamp index."""
    sensor = sensor.sort_index()
    ctd_times = ctd.index.to_numpy(dtype=float)
    sensor_times = sensor.index.to_numpy(dtype=float)
    out = pd.DataFrame(index=ctd.index, columns=sensor.columns, dtype=float)

    valid_time = np.isfinite(sensor_times)
    for col in sensor.columns:
        values = pd.to_numeric(sensor[col], errors="coerce").to_numpy(dtype=float)
        mask = valid_time & np.isfinite(values)
        if mask.sum() == 0:
            out[col] = np.nan
        elif mask.sum() == 1:
            out[col] = values[mask][0]
        else:
            out[col] = np.interp(ctd_times, sensor_times[mask], values[mask], left=np.nan, right=np.nan)
    out.index.name = "time_ms"
    return out


def combine_sensors(
    ctd: pd.DataFrame,
    absorption: pd.DataFrame | None = None,
    attenuation: pd.DataFrame | None = None,
    vsf: pd.DataFrame | None = None,
    acs_internal: pd.DataFrame | None = None,
    depth_offset_m: float = 0.0,
) -> pd.DataFrame:
    """Build one synchronized table on CTD time, with CTD pressure as depth."""
    combined = ctd.copy()
    combined.insert(0, "depth_m", pd.to_numeric(ctd["Pres(dbar)"], errors="coerce") + depth_offset_m)

    if vsf is not None and not vsf.empty:
        combined = combined.join(add_prefix(interpolate_to_ctd_time(ctd, vsf), "vsf_"))
    if attenuation is not None and not attenuation.empty:
        combined = combined.join(add_prefix(interpolate_to_ctd_time(ctd, attenuation), "c_"))
    if absorption is not None and not absorption.empty:
        combined = combined.join(add_prefix(interpolate_to_ctd_time(ctd, absorption), "a_"))
    if acs_internal is not None and not acs_internal.empty:
        combined = combined.join(add_prefix(interpolate_to_ctd_time(ctd, acs_internal), "acs_"))

    return combined


def split_profile(df: pd.DataFrame) -> ProfileSegments:
    """Split a profile at maximum depth into downcast and upcast."""
    depth = pd.to_numeric(df["depth_m"], errors="coerce")
    if depth.notna().sum() == 0:
        raise ValueError("Cannot split profile because depth_m contains no valid values")
    max_pos = int(np.nanargmax(depth.to_numpy(dtype=float)))
    down = df.iloc[: max_pos + 1].copy()
    up = df.iloc[max_pos:].copy()
    return ProfileSegments(all=df.copy(), down=down, up=up)


def sort_by_depth(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("depth_m")
    out.index.name = "time_ms"
    return out


def wavelength_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = []
    for col in df.columns:
        if not col.startswith(prefix):
            continue
        try:
            float(col.removeprefix(prefix))
        except ValueError:
            continue
        cols.append(col)
    return sorted(cols, key=lambda name: float(name.removeprefix(prefix)))


def interpolate_wavelengths(df: pd.DataFrame, prefix: str, wavelengths: np.ndarray) -> pd.DataFrame:
    cols = wavelength_columns(df, prefix)
    if not cols:
        return pd.DataFrame(index=df.index)
    source_wl = np.array([float(col.removeprefix(prefix)) for col in cols], dtype=float)
    values = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    interpolated = np.empty((len(df), len(wavelengths)), dtype=float)
    for i, row in enumerate(values):
        mask = np.isfinite(row)
        if mask.sum() < 2:
            interpolated[i, :] = np.nan
        else:
            interpolated[i, :] = np.interp(wavelengths, source_wl[mask], row[mask])
    return pd.DataFrame(interpolated, index=df.index, columns=[f"{wl:g}" for wl in wavelengths])


def absorption_line_height(
    df: pd.DataFrame,
    peak_nm: float = 676,
    min_nm: float = 650,
    max_nm: float = 715,
) -> pd.Series:
    wavelengths = np.array([min_nm, peak_nm, max_nm], dtype=float)
    interp = interpolate_wavelengths(df, "a_", wavelengths)
    if interp.empty:
        return pd.Series(index=df.index, dtype=float, name="aLH")
    y_min = interp[f"{min_nm:g}"]
    y_peak = interp[f"{peak_nm:g}"]
    y_max = interp[f"{max_nm:g}"]
    baseline = y_min + ((y_max - y_min) / (max_nm - min_nm)) * (peak_nm - min_nm)
    return (y_peak - baseline).rename("aLH")


def segment_summary(name: str, df: pd.DataFrame) -> dict[str, float | str | int]:
    depth = pd.to_numeric(df["depth_m"], errors="coerce")
    out: dict[str, float | str | int] = {
        "segment": name,
        "rows": int(len(df)),
        "min_time_ms": float(df.index.min()) if len(df) else np.nan,
        "max_time_ms": float(df.index.max()) if len(df) else np.nan,
        "min_depth_m": float(depth.min()) if depth.notna().any() else np.nan,
        "max_depth_m": float(depth.max()) if depth.notna().any() else np.nan,
    }
    for col in ["Temp(C)", "Cond(S/m)", "Sal(PSU)"]:
        if col in df:
            vals = pd.to_numeric(df[col], errors="coerce")
            out[f"median_{col}"] = float(vals.median()) if vals.notna().any() else np.nan
    alh = absorption_line_height(df)
    if alh.notna().any():
        out["median_aLH"] = float(alh.median())
        out["max_aLH"] = float(alh.max())
    return out


def trim_fraction(df: pd.DataFrame, trim_percent: float) -> pd.DataFrame:
    if trim_percent <= 0:
        return df.copy()
    n_trim = int(round(len(df) * trim_percent / 100.0))
    if n_trim == 0:
        return df.copy()
    if 2 * n_trim >= len(df):
        raise ValueError("trim_percent removes all lab records")
    return df.iloc[n_trim:-n_trim].copy()


def lab_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = pd.DataFrame(
        {
            "mean": df.mean(axis=0, numeric_only=True),
            "median": df.median(axis=0, numeric_only=True),
            "std": df.std(axis=0, numeric_only=True),
            "q25": df.quantile(0.25, axis=0, numeric_only=True),
            "q75": df.quantile(0.75, axis=0, numeric_only=True),
        }
    )
    stats.index.name = "wavelength_nm"
    return stats

