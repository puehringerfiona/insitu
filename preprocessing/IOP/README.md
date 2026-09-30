# IOP Probe Processing
# Abolfazl Irani Rahaghi (irani.abolfazl@gmail.com)

Function-based Python scripts for reading, synchronizing, exporting, plotting, and dashboarding IOP optical probe data.

## Quick Start

Profile cast with ACS, VSF, and CTD:

```powershell
python run_iop_probe.py profile `
  "C:\Users\airani\Documents\supervisions\Fiona\IOP_data_Fiona\extracted\20240618_BIE_IOP_extracted" `
  --run 292
```

Clean-water ACS lab measurement:

```powershell
python run_iop_probe.py lab `
  "C:\path\to\extracted\folder" `
  --run 139 `
  --trim-percent 30
```

Outputs are written to a new folder beside the input by default:

```text
IOP_outputs/<input-folder-name>_run_<run>/
  csv/
  plots/
  dashboard.html
```

Use `--output-dir` to choose another location.

## Windows App

For users who do not want to run Python commands, build the Windows app and share the resulting `dist/IOPProbeProcessor` folder:

```powershell
python -m pip install -r requirements.txt pyinstaller
python build_windows_exe.py
```

The executable is created at:

```text
dist/IOPProbeProcessor/IOPProbeProcessor.exe
```

Users can copy the full `IOPProbeProcessor` folder to the field laptop, double-click `IOPProbeProcessor.exe`, select the extracted measurement folder, enter the run number, and click **Process Measurements**.

For profile casts, the `csv/` folder contains both raw synchronized exports and ACS-QC exports:

```text
synchronized_all.csv
synchronized_down.csv
synchronized_up.csv
synchronized_acs_qc_all.csv
synchronized_acs_qc_down.csv
synchronized_acs_qc_up.csv
acs_qc_flags_all.csv
acs_qc_flags_down.csv
acs_qc_flags_up.csv
summary.csv
acs_qc_summary.csv
```

For lab measurements, the raw trimmed ACS files and statistics are preserved, and ACS-QC versions are added:

```text
lab_absorption_trimmed.csv
lab_attenuation_trimmed.csv
lab_absorption_stats.csv
lab_attenuation_stats.csv
lab_absorption_acs_qc_trimmed.csv
lab_attenuation_acs_qc_trimmed.csv
lab_absorption_acs_qc_stats.csv
lab_attenuation_acs_qc_stats.csv
lab_acs_qc_flags.csv
lab_acs_qc_summary.csv
```

## Notes

- The ACS parser detects the `Time(ms)` data header instead of relying on a fixed row number.
- Sensor timestamps are sorted and duplicate timestamps are averaged before interpolation.
- CTD pressure is treated as depth in meters by default (`depth_m = Pres(dbar)`). Use `--depth-offset-m` if you want to shift plotted ACS depth to account for sensor geometry.
- Profile data are split into downcast and upcast at the maximum CTD pressure.
- The synchronized CSVs combine CTD, VSF, ACS absorption (`a_...`) and ACS attenuation (`c_...`) columns on the CTD time grid.
- ACS QC is implemented in `iop_probe/qc.py`. It removes/flags NaNs, negative values, rolling-MAD spikes, problematic absorption spectral shapes, and wavelengths above 720 nm; then it interpolates small gaps, applies a safer green-band offset correction to absorption, and smooths along the profile with a centered rolling median.
