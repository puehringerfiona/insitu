from __future__ import annotations

from pathlib import Path
import os

import pandas as pd

from .dashboard import write_dashboard
from .io import add_prefix, find_run_file, infer_output_dir, read_acs, read_ctd, read_vsf, write_csv
from .plotting import (
    plot_acs_heatmap,
    plot_alh_temperature,
    plot_depth_spectra,
    plot_lab_spectra,
    plot_lab_timeseries,
    plot_profile_overview,
)
from .processing import combine_sensors, lab_stats, segment_summary, sort_by_depth, split_profile, trim_fraction, wavelength_columns
from .qc import apply_acs_qc


def _safe_plot(plotter, *args, errors: list[str], **kwargs):
    try:
        return plotter(*args, **kwargs)
    except ImportError as exc:
        errors.append(f"Plotting dependency missing: {exc}")
    except Exception as exc:
        errors.append(f"{plotter.__name__} failed: {exc}")
    return None


def _set_matplotlib_cache(output_dir: Path) -> None:
    cache_dir = output_dir / ".matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)


def run_profile(
    input_dir: str | Path,
    run: str | int,
    output_dir: str | Path | None = None,
    depth_offset_m: float = 0.0,
) -> dict[str, object]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else infer_output_dir(input_dir, run)
    csv_dir = output_dir / "csv"
    plot_dir = output_dir / "plots"
    _set_matplotlib_cache(output_dir)
    errors: list[str] = []

    acs_path = find_run_file(input_dir, run, "ACS")
    vsf_path = find_run_file(input_dir, run, "MISC-ASCII")
    ctd_path = find_run_file(input_dir, run, "CTD-ENGR")

    absorption, attenuation, acs_internal = read_acs(acs_path)
    vsf = read_vsf(vsf_path)
    ctd = read_ctd(ctd_path)

    synchronized = combine_sensors(ctd, absorption, attenuation, vsf, acs_internal, depth_offset_m)
    segments = split_profile(synchronized)
    segment_map = {
        "all": segments.all,
        "down": sort_by_depth(segments.down),
        "up": sort_by_depth(segments.up),
    }
    qc_results = {name: apply_acs_qc(df) for name, df in segment_map.items()}
    qc_segment_map = {name: result.data for name, result in qc_results.items()}

    csv_files: list[Path] = []
    for name, df in segment_map.items():
        path = csv_dir / f"synchronized_{name}.csv"
        write_csv(df, path)
        csv_files.append(path)
    for name, result in qc_results.items():
        data_path = csv_dir / f"synchronized_acs_qc_{name}.csv"
        flag_path = csv_dir / f"acs_qc_flags_{name}.csv"
        write_csv(result.data, data_path)
        write_csv(result.flags, flag_path)
        csv_files.extend([data_path, flag_path])

    summary_raw = pd.DataFrame([segment_summary(name, df) | {"processing": "raw"} for name, df in segment_map.items()])
    summary_qc = pd.DataFrame([segment_summary(name, df) | {"processing": "acs_qc"} for name, df in qc_segment_map.items()])
    summary = pd.concat([summary_raw, summary_qc], ignore_index=True)
    summary_path = csv_dir / "summary.csv"
    write_csv(summary.set_index(["processing", "segment"]), summary_path)
    csv_files.append(summary_path)
    qc_summary = pd.concat(
        [result.summary.assign(segment=name) for name, result in qc_results.items()],
        ignore_index=True,
    )
    qc_summary_path = csv_dir / "acs_qc_summary.csv"
    write_csv(qc_summary.set_index(["segment", "channel"]), qc_summary_path)
    csv_files.append(qc_summary_path)

    plot_files: list[Path] = []
    title = f"{input_dir.name} run {run}"
    maybe = _safe_plot(plot_profile_overview, segment_map, plot_dir / "profile_overview.png", title, errors=errors)
    if maybe:
        plot_files.append(maybe)
    maybe = _safe_plot(plot_depth_spectra, {"down": qc_segment_map["down"], "up": qc_segment_map["up"]}, plot_dir / "acs_qc_spectra_down_up.png", f"{title} ACS QC", errors=errors)
    if maybe:
        plot_files.append(maybe)
    maybe = _safe_plot(plot_alh_temperature, {"down": qc_segment_map["down"], "up": qc_segment_map["up"]}, plot_dir / "acs_qc_alh_temperature_down_up.png", f"{title} ACS QC", errors=errors)
    if maybe:
        plot_files.append(maybe)
    for name in ["down", "up"]:
        maybe = _safe_plot(plot_acs_heatmap, qc_segment_map[name], plot_dir / f"acs_qc_heatmap_{name}.png", f"{title} {name} ACS QC", errors=errors)
        if maybe:
            plot_files.append(maybe)
    maybe = _safe_plot(plot_depth_spectra, {"down": segment_map["down"], "up": segment_map["up"]}, plot_dir / "raw_acs_spectra_down_up.png", f"{title} raw", errors=errors)
    if maybe:
        plot_files.append(maybe)

    dashboard = write_dashboard(output_dir / "dashboard.html", title, summary, csv_files, plot_files)

    return {
        "output_dir": output_dir,
        "csv_files": csv_files,
        "plot_files": plot_files,
        "dashboard": dashboard,
        "errors": errors,
    }


def run_lab(
    input_dir: str | Path,
    run: str | int,
    output_dir: str | Path | None = None,
    trim_percent: float = 30.0,
) -> dict[str, object]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else infer_output_dir(input_dir, run)
    csv_dir = output_dir / "csv"
    plot_dir = output_dir / "plots"
    _set_matplotlib_cache(output_dir)
    errors: list[str] = []

    acs_path = find_run_file(input_dir, run, "ACS")
    absorption, attenuation, acs_internal = read_acs(acs_path)
    absorption = trim_fraction(absorption, trim_percent)
    attenuation = trim_fraction(attenuation, trim_percent)
    acs_internal = trim_fraction(acs_internal, trim_percent) if not acs_internal.empty else acs_internal

    a_stats = lab_stats(absorption)
    c_stats = lab_stats(attenuation)
    lab_qc_input = add_prefix(absorption, "a_").join(add_prefix(attenuation, "c_"))
    lab_qc_result = apply_acs_qc(lab_qc_input)
    a_qc_cols = wavelength_columns(lab_qc_result.data, "a_")
    c_qc_cols = wavelength_columns(lab_qc_result.data, "c_")
    absorption_qc = lab_qc_result.data[a_qc_cols].copy()
    attenuation_qc = lab_qc_result.data[c_qc_cols].copy()
    absorption_qc.columns = [float(col.removeprefix("a_")) for col in absorption_qc.columns]
    attenuation_qc.columns = [float(col.removeprefix("c_")) for col in attenuation_qc.columns]
    a_qc_stats = lab_stats(absorption_qc)
    c_qc_stats = lab_stats(attenuation_qc)

    csv_files = [
        csv_dir / "lab_absorption_trimmed.csv",
        csv_dir / "lab_attenuation_trimmed.csv",
        csv_dir / "lab_absorption_stats.csv",
        csv_dir / "lab_attenuation_stats.csv",
        csv_dir / "lab_absorption_acs_qc_trimmed.csv",
        csv_dir / "lab_attenuation_acs_qc_trimmed.csv",
        csv_dir / "lab_absorption_acs_qc_stats.csv",
        csv_dir / "lab_attenuation_acs_qc_stats.csv",
        csv_dir / "lab_acs_qc_flags.csv",
    ]
    write_csv(absorption, csv_files[0])
    write_csv(attenuation, csv_files[1])
    write_csv(a_stats, csv_files[2])
    write_csv(c_stats, csv_files[3])
    write_csv(absorption_qc, csv_files[4])
    write_csv(attenuation_qc, csv_files[5])
    write_csv(a_qc_stats, csv_files[6])
    write_csv(c_qc_stats, csv_files[7])
    write_csv(lab_qc_result.flags, csv_files[8])
    if not acs_internal.empty:
        internal_path = csv_dir / "lab_acs_internal_trimmed.csv"
        write_csv(acs_internal, internal_path)
        csv_files.append(internal_path)

    summary = pd.DataFrame(
        [
            {
                "processing": "raw",
                "segment": "lab_trimmed",
                "rows": len(absorption),
                "trim_percent_each_end": trim_percent,
                "min_time_ms": float(absorption.index.min()),
                "max_time_ms": float(absorption.index.max()),
                "median_a_676": float(a_stats.loc[min(a_stats.index, key=lambda x: abs(float(x) - 676)), "median"]),
                "median_c_676": float(c_stats.loc[min(c_stats.index, key=lambda x: abs(float(x) - 676)), "median"]),
            },
            {
                "processing": "acs_qc",
                "segment": "lab_trimmed",
                "rows": len(absorption_qc),
                "trim_percent_each_end": trim_percent,
                "min_time_ms": float(absorption_qc.index.min()),
                "max_time_ms": float(absorption_qc.index.max()),
                "median_a_676": float(a_qc_stats.loc[min(a_qc_stats.index, key=lambda x: abs(float(x) - 676)), "median"]),
                "median_c_676": float(c_qc_stats.loc[min(c_qc_stats.index, key=lambda x: abs(float(x) - 676)), "median"]),
            },
        ]
    )
    summary_path = csv_dir / "summary.csv"
    write_csv(summary.set_index(["processing", "segment"]), summary_path)
    csv_files.append(summary_path)
    qc_summary_path = csv_dir / "lab_acs_qc_summary.csv"
    write_csv(lab_qc_result.summary.assign(segment="lab_trimmed").set_index(["segment", "channel"]), qc_summary_path)
    csv_files.append(qc_summary_path)

    title = f"{input_dir.name} run {run} lab"
    plot_files: list[Path] = []
    maybe = _safe_plot(plot_lab_spectra, a_stats, c_stats, plot_dir / "lab_spectra_stats.png", title, errors=errors)
    if maybe:
        plot_files.append(maybe)
    maybe = _safe_plot(plot_lab_spectra, a_qc_stats, c_qc_stats, plot_dir / "lab_acs_qc_spectra_stats.png", f"{title} ACS QC", errors=errors)
    if maybe:
        plot_files.append(maybe)
    maybe = _safe_plot(plot_lab_timeseries, absorption, attenuation, plot_dir / "lab_selected_wavelength_timeseries.png", title, errors=errors)
    if maybe:
        plot_files.append(maybe)
    maybe = _safe_plot(plot_lab_timeseries, absorption_qc, attenuation_qc, plot_dir / "lab_acs_qc_selected_wavelength_timeseries.png", f"{title} ACS QC", errors=errors)
    if maybe:
        plot_files.append(maybe)

    dashboard = write_dashboard(output_dir / "dashboard.html", title, summary, csv_files, plot_files)

    return {
        "output_dir": output_dir,
        "csv_files": csv_files,
        "plot_files": plot_files,
        "dashboard": dashboard,
        "errors": errors,
    }
