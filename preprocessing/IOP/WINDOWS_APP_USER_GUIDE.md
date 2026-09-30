# IOP Probe Processor User Guide

## What This App Does

The app processes extracted IOP probe measurement folders and creates:

- synchronized CSV files
- ACS-QC CSV files and QC flags
- high-resolution PNG plots
- one `dashboard.html` file with all plots and output links

## How To Use

1. Open the `IOPProbeProcessor` folder.
2. Double-click `IOPProbeProcessor.exe`.
3. Click **Browse** next to **Measurement folder** and select the extracted run folder.
4. Enter the run number, for example `292`.
5. Choose the measurement type:
   - **Vertical profile** for field casts with ACS, VSF, and CTD.
   - **Clean-water lab** for ACS-only clean-water measurements.
6. Leave **Output folder** blank unless you want to choose a specific output location.
7. Click **Process Measurements**.
8. When finished, click **Open Output Folder** or **Open Dashboard**.

## Output Location

If no output folder is selected, the app creates:

```text
IOP_outputs/<measurement-folder-name>_run_<run-number>/
```

inside the parent folder of the selected measurement folder.

## Files Needed In The Measurement Folder

For vertical profiles:

```text
run_21_ACS.<run-number>
run_22_MISC-ASCII.<run-number>
run_23_CTD-ENGR.<run-number>
```

For clean-water lab measurements:

```text
run_21_ACS.<run-number>
```

## Notes

- Keep the full `IOPProbeProcessor` folder together. Do not move only the `.exe` file by itself.
- The first run may take a little longer while Windows checks the new application.
- If Windows SmartScreen appears, choose **More info** and then **Run anyway** if you trust this local build.

