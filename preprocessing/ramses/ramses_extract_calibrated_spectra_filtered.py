"""
Extract calibrated RAMSES spectra for Ed, Lu, and Ls/Ld sensors.

The script reads calibrated .dat files, reshapes spectra into wavelength-by-time
tables, and keeps only timestamps shared by all three radiometric quantities.
Defaults below can be edited for the current campaign; command-line arguments
override them.
"""

import glob
import os
import re
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


# Default campaign settings. Override these with CLI arguments when needed.
DEFAULT_INPUT_DIRECTORY = Path(r"C:\Users\puehrifi\Documents\insitu\20260430_ZRH\acquisition\AOP")
DEFAULT_OUTPUT_DIRECTORY = Path(
    r"C:\Users\puehrifi\Documents\insitu\20260430_ZRH\preprocessing\20260430_ZRH_ramses_spectra"
)
DEFAULT_ED_DEVICE = "SAM_8622"
DEFAULT_LU_DEVICE = "SAM_8761"
DEFAULT_LS_DEVICE = "SAM_8481"


def parse_args():
    parser = ArgumentParser(
        description=(
            "Extract calibrated RAMSES spectra and keep only timestamps shared "
            "by Ed, Lu, and Ls/Ld devices."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--ed-device", default=DEFAULT_ED_DEVICE)
    parser.add_argument("--lu-device", default=DEFAULT_LU_DEVICE)
    parser.add_argument("--ls-device", default=DEFAULT_LS_DEVICE)
    return parser.parse_args()


def parse_dat_files(input_dir, devices):
    spectrum_pattern = re.compile(r"\[Spectrum\]", re.IGNORECASE)
    device_data = {device: [] for device in devices}

    for filepath in glob.glob(os.path.join(str(input_dir), "*.dat")):
        with open(filepath, "r", encoding="utf-8") as file:
            lines = file.readlines()

        spectrum_data = []
        current_device = None
        current_datetime = None
        recording = False

        for i, line in enumerate(lines):
            line = line.strip()

            if spectrum_pattern.match(line):
                recording = True
                spectrum_data = []
                current_device = None
                current_datetime = None

            if recording:
                if line.startswith("IDDevice"):
                    current_device = line.split("=")[-1].strip()
                elif line.startswith("DateTime"):
                    current_datetime = line.split("=")[-1].strip()
                elif line.startswith("IDDataTypeSub1") and "CALIBRATED" not in line:
                    recording = False
                elif line.startswith("[DATA]"):
                    for data_line in lines[i + 1:]:
                        data_line = data_line.strip()
                        if data_line.startswith("[END] of [DATA]"):
                            break
                        parts = data_line.split()
                        if len(parts) >= 2:
                            try:
                                wavelength = float(parts[0])
                                intensity = float(parts[1])
                                spectrum_data.append((wavelength, current_datetime, intensity))
                            except ValueError:
                                continue

            if line.startswith("[END] of Spectrum"):
                recording = False
                if current_device in device_data:
                    device_data[current_device].extend(spectrum_data)

    return device_data


def reshape_device_data(device_data):
    pivot_tables = {}
    for device, data in device_data.items():
        if data:
            df = pd.DataFrame(data, columns=["Wavelength (nm)", "DateTime", "Intensity"])
            df = df.groupby(["Wavelength (nm)", "DateTime"], as_index=False)["Intensity"].mean()
            df_pivot = df.pivot(index="Wavelength (nm)", columns="DateTime", values="Intensity")
            pivot_tables[device] = df_pivot
    return pivot_tables


def get_common_columns(dfs):
    if not dfs:
        return []
    common = set(dfs[0].columns)
    for df in dfs[1:]:
        common &= set(df.columns)
    return sorted(common)


def save_filtered_tables(pivot_tables, common_columns, output_dir, device_roles):
    for device, role in device_roles.items():
        filtered_df = pivot_tables[device][common_columns]
        filename = f"{role}_reshaped_spectrum_data_filtered.csv"
        filtered_df.to_csv(os.path.join(output_dir, filename))


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device_roles = {
        args.ed_device: "Ed",
        args.lu_device: "Lu",
        args.ls_device: "Ls",
    }
    expected_devices = list(device_roles)

    device_data = parse_dat_files(args.input_dir, expected_devices)
    pivot_tables = reshape_device_data(device_data)

    missing = [dev for dev in expected_devices if dev not in pivot_tables]
    if missing:
        raise SystemExit(f"Missing expected device spectra: {', '.join(missing)}")

    dfs = [pivot_tables[dev] for dev in expected_devices]
    common_columns = get_common_columns(dfs)
    if not common_columns:
        raise SystemExit("No shared timestamps found across Ed, Lu, and Ls devices.")

    save_filtered_tables(pivot_tables, common_columns, args.output_dir, device_roles)
    print(f"Saved filtered spectra for {len(common_columns)} shared timestamps to: {args.output_dir}")


if __name__ == "__main__":
    main()
