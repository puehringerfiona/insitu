from __future__ import annotations

from pathlib import Path
import os
import sys
import tempfile

import numpy as np
import pandas as pd

from .processing import absorption_line_height, wavelength_columns


def _mpl():
    project_dir = Path(__file__).resolve().parents[1]
    if os.environ.get("MPLCONFIGDIR"):
        mpl_config_dir = Path(os.environ["MPLCONFIGDIR"])
    elif getattr(sys, "frozen", False):
        mpl_config_dir = Path(tempfile.gettempdir()) / "IOPProbeProcessor_matplotlib"
    else:
        mpl_config_dir = project_dir / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
    for vendor_dir in [project_dir / ".codex_mpl", project_dir / ".codex_pydeps"]:
        if vendor_dir.exists() and str(vendor_dir) not in sys.path:
            sys.path.append(str(vendor_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: Path, dpi: int = 300) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    _mpl().close(fig)
    return path


def _depth_axis(df: pd.DataFrame) -> np.ndarray:
    return -pd.to_numeric(df["depth_m"], errors="coerce").to_numpy(dtype=float)


def plot_profile_overview(segments: dict[str, pd.DataFrame], output: Path, title: str) -> Path:
    plt = _mpl()
    fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)
    styles = {"down": "-", "up": "--", "all": ":"}
    colors = {"down": "#1f77b4", "up": "#d62728", "all": "#666666"}

    profile_vars = [
        ("Temp(C)", "Temperature (C)"),
        ("Cond(S/m)", "Conductivity (S/m)"),
        ("Sal(PSU)", "Salinity (PSU)"),
        ("vsf_G_125", "VSF G 125"),
    ]
    for ax, (col, label) in zip(axes, profile_vars):
        for name in ["down", "up"]:
            df = segments.get(name)
            if df is not None and col in df:
                ax.plot(df[col], _depth_axis(df), linestyle=styles[name], color=colors[name], label=name)
        ax.set_xlabel(label)
        ax.grid(True, color="#dddddd", linewidth=0.6)
    axes[0].set_ylabel("Depth (m)")
    axes[0].legend(frameon=False)
    fig.suptitle(title)
    return _save(fig, output)


def plot_acs_heatmap(df: pd.DataFrame, output: Path, title: str) -> Path | None:
    a_cols = wavelength_columns(df, "a_")
    c_cols = wavelength_columns(df, "c_")
    if not a_cols or not c_cols:
        return None

    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), sharey=True)
    depth = _depth_axis(df)
    for ax, cols, label, use_log in [
        (axes[0], a_cols, "log10 absorption a", True),
        (axes[1], c_cols, "attenuation c", False),
    ]:
        wl = np.array([float(col.split("_", 1)[1]) for col in cols])
        values = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if use_log:
            values = np.log10(np.where(values > 0, values, np.nan))
        mesh = ax.pcolormesh(wl, depth, values, shading="auto", cmap="viridis")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_title(label)
        fig.colorbar(mesh, ax=ax, shrink=0.88)
    axes[0].set_ylabel("Depth (m)")
    fig.suptitle(title)
    return _save(fig, output)


def plot_depth_spectra(segments: dict[str, pd.DataFrame], output: Path, title: str) -> Path | None:
    first = next((df for df in segments.values() if df is not None and not df.empty), None)
    if first is None:
        return None
    a_cols = wavelength_columns(first, "a_")
    c_cols = wavelength_columns(first, "c_")
    if not a_cols or not c_cols:
        return None

    max_depth = max(float(pd.to_numeric(df["depth_m"], errors="coerce").max()) for df in segments.values() if len(df))
    target_depths = [d for d in [1, 4, 7, 11, 15, 20] if d <= max_depth + 0.5]
    if not target_depths:
        target_depths = [max_depth]

    plt = _mpl()
    fig, axes = plt.subplots(2, 1, figsize=(9, 11), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(target_depths)))
    styles = {"down": "-", "up": "--"}

    for seg_name, df in segments.items():
        if seg_name not in styles or df is None or df.empty:
            continue
        depth = pd.to_numeric(df["depth_m"], errors="coerce").to_numpy(dtype=float)
        for color, target in zip(colors, target_depths):
            idx = int(np.nanargmin(np.abs(depth - target)))
            a_wl = [float(col.split("_", 1)[1]) for col in a_cols]
            c_wl = [float(col.split("_", 1)[1]) for col in c_cols]
            axes[0].plot(a_wl, df.iloc[idx][a_cols], linestyle=styles[seg_name], color=color, label=f"{target:g} m {seg_name}")
            axes[1].plot(c_wl, df.iloc[idx][c_cols], linestyle=styles[seg_name], color=color, label=f"{target:g} m {seg_name}")
    axes[0].set_ylabel("Absorption a (1/m)")
    axes[1].set_ylabel("Attenuation c (1/m)")
    axes[1].set_xlabel("Wavelength (nm)")
    for ax in axes:
        ax.grid(True, color="#dddddd", linewidth=0.6)
    axes[0].legend(ncol=2, fontsize=8, frameon=False)
    fig.suptitle(title)
    return _save(fig, output)


def plot_alh_temperature(segments: dict[str, pd.DataFrame], output: Path, title: str) -> Path | None:
    if not any(wavelength_columns(df, "a_") for df in segments.values() if df is not None):
        return None
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    styles = {"down": "-", "up": "--"}
    colors = {"down": "#1f77b4", "up": "#d62728"}
    for name, df in segments.items():
        if name not in styles or df is None or df.empty:
            continue
        depth = _depth_axis(df)
        if "Temp(C)" in df:
            axes[0].plot(df["Temp(C)"], depth, linestyle=styles[name], color=colors[name], label=name)
        alh = absorption_line_height(df)
        axes[1].plot(alh, depth, linestyle=styles[name], color=colors[name], label=name)
    axes[0].set_xlabel("Temperature (C)")
    axes[0].set_ylabel("Depth (m)")
    axes[1].set_xlabel("aLH 676 over 650-715 nm (1/m)")
    for ax in axes:
        ax.grid(True, color="#dddddd", linewidth=0.6)
        ax.legend(frameon=False)
    fig.suptitle(title)
    return _save(fig, output)


def plot_lab_spectra(a_stats: pd.DataFrame, c_stats: pd.DataFrame, output: Path, title: str) -> Path:
    plt = _mpl()
    fig, axes = plt.subplots(2, 1, figsize=(9, 11), sharex=True)
    for ax, stats, ylabel, color in [
        (axes[0], a_stats, "Absorption a (1/m)", "#1f77b4"),
        (axes[1], c_stats, "Attenuation c (1/m)", "#2ca02c"),
    ]:
        wl = stats.index.to_numpy(dtype=float)
        ax.plot(wl, stats["median"], color=color, linewidth=2, label="median")
        ax.fill_between(wl, stats["q25"], stats["q75"], color=color, alpha=0.22, label="25-75%")
        ax.plot(wl, stats["mean"], color="#222222", linewidth=1, alpha=0.8, label="mean")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#dddddd", linewidth=0.6)
        ax.legend(frameon=False)
    axes[1].set_xlabel("Wavelength (nm)")
    fig.suptitle(title)
    return _save(fig, output)


def plot_lab_timeseries(absorption: pd.DataFrame, attenuation: pd.DataFrame, output: Path, title: str) -> Path:
    plt = _mpl()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, df, ylabel in [
        (axes[0], absorption, "Absorption a (1/m)"),
        (axes[1], attenuation, "Attenuation c (1/m)"),
    ]:
        cols = list(df.columns)
        targets = [440, 532, 676, 715]
        for target in targets:
            col = min(cols, key=lambda c: abs(float(c) - target))
            ax.plot(df.index / 1000, df[col], label=f"{float(col):g} nm")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#dddddd", linewidth=0.6)
        ax.legend(frameon=False, ncol=4)
    axes[1].set_xlabel("Elapsed time (s)")
    fig.suptitle(title)
    return _save(fig, output)
