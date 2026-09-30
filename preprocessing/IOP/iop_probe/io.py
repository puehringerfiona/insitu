from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


VSF_COLUMNS = [
    "B_100",
    "B_125",
    "B_150",
    "G_100",
    "G_125",
    "G_150",
    "R_100",
    "R_125",
    "R_150",
]

CTD_COLUMNS = ["Pres(dbar)", "Temp(C)", "Cond(S/m)", "Sal(PSU)"]


def find_run_file(input_dir: Path, run: str | int, token: str) -> Path:
    """Find a file such as run_21_ACS.292 within an extracted run folder."""
    matches = sorted(input_dir.glob(f"*{token}*.{run}"))
    if not matches:
        raise FileNotFoundError(f"Could not find *{token}*.{run} in {input_dir}")
    return matches[0]


def _normalise_time_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_numeric(df.index, errors="coerce")
    df = df.loc[df.index.notna()]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.sort_index()
    if df.index.has_duplicates:
        df = df.groupby(level=0).mean(numeric_only=True)
    df.index.name = "time_ms"
    return df


def _clean_sensor_columns(columns: list[str]) -> list[str]:
    return [col.strip().strip("_") for col in columns]


def read_acs(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read ACS data and return absorption, attenuation, and internal columns."""
    lines = path.read_text(errors="replace").splitlines()
    header_idx = next(
        (idx for idx, line in enumerate(lines) if line.strip().lower().startswith("time(ms)")),
        None,
    )
    if header_idx is None:
        raise ValueError(f"Could not find ACS data header beginning with Time(ms): {path}")

    header = lines[header_idx].split()
    rows: list[list[float]] = []
    for line in lines[header_idx + 1 :]:
        parts = line.split()
        if len(parts) != len(header):
            continue
        try:
            rows.append([float(value) for value in parts])
        except ValueError:
            continue

    if not rows:
        raise ValueError(f"No numeric ACS rows found in {path}")

    df = pd.DataFrame(rows, columns=header).set_index("Time(ms)")
    df = _normalise_time_index(df)

    c_cols = [col for col in df.columns if re.fullmatch(r"[cC]\d+(?:\.\d+)?", col)]
    a_cols = [col for col in df.columns if re.fullmatch(r"[aA]\d+(?:\.\d+)?", col)]
    internal_cols = [col for col in df.columns if col not in c_cols and col not in a_cols]

    attenuation = df[c_cols].copy()
    absorption = df[a_cols].copy()
    internal = df[internal_cols].copy()

    attenuation.columns = [float(col[1:]) for col in attenuation.columns]
    absorption.columns = [float(col[1:]) for col in absorption.columns]
    internal.columns = _clean_sensor_columns(list(internal.columns))

    return absorption, attenuation, internal


def read_vsf(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    first = df.columns[0]
    df = df.set_index(first)
    df = _normalise_time_index(df)
    df.columns = VSF_COLUMNS[: len(df.columns)]
    return df


def read_ctd(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    first = df.columns[0]
    df = df.set_index(first)
    df = _normalise_time_index(df)
    if len(df.columns) >= len(CTD_COLUMNS):
        df = df.iloc[:, : len(CTD_COLUMNS)]
        df.columns = CTD_COLUMNS
    return df


def add_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    out.columns = [f"{prefix}{col:g}" if isinstance(col, float) else f"{prefix}{col}" for col in out.columns]
    return out


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True, na_rep="NaN")


def infer_output_dir(input_dir: Path, run: str | int) -> Path:
    return input_dir.parent / "IOP_outputs" / f"{input_dir.name}_run_{run}"


def finite_depth_summary(ctd: pd.DataFrame) -> dict[str, float]:
    pressure = pd.to_numeric(ctd["Pres(dbar)"], errors="coerce").to_numpy(dtype=float)
    pressure = pressure[np.isfinite(pressure)]
    if pressure.size == 0:
        return {"min_depth_m": np.nan, "max_depth_m": np.nan}
    return {"min_depth_m": float(np.nanmin(pressure)), "max_depth_m": float(np.nanmax(pressure))}
