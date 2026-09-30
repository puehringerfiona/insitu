from __future__ import annotations

import os
import queue
import json
import argparse
import threading
import traceback
import webbrowser
from pathlib import Path
import sys
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk

from iop_probe import run_lab, run_profile


APP_TITLE = "IOP Probe Processor"


class IOPProbeApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.minsize(760, 620)

        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.run_number = tk.StringVar()
        self.mode = tk.StringVar(value="profile")
        self.depth_offset = tk.StringVar(value="0.0")
        self.trim_percent = tk.StringVar(value="30")
        self.status = tk.StringVar(value="Ready")

        self._events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._last_output_dir: Path | None = None
        self._last_dashboard: Path | None = None

        self._build_ui()
        self._refresh_mode_fields()
        self.after(120, self._drain_events)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=18)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(5, weight=1)

        header = ttk.Label(root, text=APP_TITLE, font=("Segoe UI", 18, "bold"))
        header.grid(row=0, column=0, sticky="w")

        subtitle = ttk.Label(
            root,
            text="Select an extracted measurement folder, enter the run number, and create CSV, PNG, and HTML outputs.",
            foreground="#4d5963",
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(4, 18))

        form = ttk.Frame(root)
        form.grid(row=2, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Measurement folder").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=6)
        ttk.Entry(form, textvariable=self.input_dir).grid(row=0, column=1, sticky="ew", pady=6)
        ttk.Button(form, text="Browse", command=self._browse_input).grid(row=0, column=2, padx=(10, 0), pady=6)

        ttk.Label(form, text="Run number").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=6)
        ttk.Entry(form, textvariable=self.run_number, width=18).grid(row=1, column=1, sticky="w", pady=6)

        ttk.Label(form, text="Measurement type").grid(row=2, column=0, sticky="w", padx=(0, 10), pady=6)
        mode_frame = ttk.Frame(form)
        mode_frame.grid(row=2, column=1, sticky="w", pady=6)
        ttk.Radiobutton(mode_frame, text="Vertical profile", variable=self.mode, value="profile", command=self._refresh_mode_fields).pack(side="left", padx=(0, 16))
        ttk.Radiobutton(mode_frame, text="Clean-water lab", variable=self.mode, value="lab", command=self._refresh_mode_fields).pack(side="left")

        self.depth_label = ttk.Label(form, text="Depth offset (m)")
        self.depth_entry = ttk.Entry(form, textvariable=self.depth_offset, width=18)
        self.depth_label.grid(row=3, column=0, sticky="w", padx=(0, 10), pady=6)
        self.depth_entry.grid(row=3, column=1, sticky="w", pady=6)

        self.trim_label = ttk.Label(form, text="Trim each end (%)")
        self.trim_entry = ttk.Entry(form, textvariable=self.trim_percent, width=18)
        self.trim_label.grid(row=4, column=0, sticky="w", padx=(0, 10), pady=6)
        self.trim_entry.grid(row=4, column=1, sticky="w", pady=6)

        ttk.Label(form, text="Output folder").grid(row=5, column=0, sticky="w", padx=(0, 10), pady=6)
        ttk.Entry(form, textvariable=self.output_dir).grid(row=5, column=1, sticky="ew", pady=6)
        ttk.Button(form, text="Browse", command=self._browse_output).grid(row=5, column=2, padx=(10, 0), pady=6)

        hint = ttk.Label(
            form,
            text="Leave output folder blank to create IOP_outputs beside the measurement folder.",
            foreground="#5f6b75",
        )
        hint.grid(row=6, column=1, sticky="w", pady=(0, 10))

        actions = ttk.Frame(root)
        actions.grid(row=3, column=0, sticky="ew", pady=(16, 10))
        actions.columnconfigure(4, weight=1)

        self.process_button = ttk.Button(actions, text="Process Measurements", command=self._start_processing)
        self.process_button.grid(row=0, column=0, padx=(0, 10))
        self.open_folder_button = ttk.Button(actions, text="Open Output Folder", command=self._open_output_folder, state="disabled")
        self.open_folder_button.grid(row=0, column=1, padx=(0, 10))
        self.open_dashboard_button = ttk.Button(actions, text="Open Dashboard", command=self._open_dashboard, state="disabled")
        self.open_dashboard_button.grid(row=0, column=2)

        ttk.Label(root, textvariable=self.status, foreground="#1f5d7a").grid(row=4, column=0, sticky="w", pady=(0, 8))

        log_frame = ttk.Frame(root)
        log_frame.grid(row=5, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = tk.Text(log_frame, wrap="word", height=14, borderwidth=1, relief="solid")
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)

    def _refresh_mode_fields(self) -> None:
        if self.mode.get() == "profile":
            self.depth_label.state(["!disabled"])
            self.depth_entry.state(["!disabled"])
            self.trim_label.state(["disabled"])
            self.trim_entry.state(["disabled"])
        else:
            self.depth_label.state(["disabled"])
            self.depth_entry.state(["disabled"])
            self.trim_label.state(["!disabled"])
            self.trim_entry.state(["!disabled"])

    def _browse_input(self) -> None:
        selected = filedialog.askdirectory(title="Select extracted measurement folder")
        if selected:
            self.input_dir.set(selected)

    def _browse_output(self) -> None:
        selected = filedialog.askdirectory(title="Select output folder")
        if selected:
            self.output_dir.set(selected)

    def _start_processing(self) -> None:
        try:
            params = self._validate_inputs()
        except ValueError as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return

        self.process_button.state(["disabled"])
        self.open_folder_button.state(["disabled"])
        self.open_dashboard_button.state(["disabled"])
        self._last_output_dir = None
        self._last_dashboard = None
        self.log.delete("1.0", "end")
        self._append_log("Starting processing...")
        self.status.set("Processing...")

        worker = threading.Thread(target=self._run_pipeline, args=(params,), daemon=True)
        worker.start()

    def _validate_inputs(self) -> dict[str, object]:
        input_dir = Path(self.input_dir.get().strip())
        if not input_dir.exists() or not input_dir.is_dir():
            raise ValueError("Please select a valid measurement folder.")

        run = self.run_number.get().strip()
        if not run:
            raise ValueError("Please enter the run number.")

        output = self.output_dir.get().strip()
        output_dir = Path(output) if output else None

        if self.mode.get() == "profile":
            try:
                depth_offset = float(self.depth_offset.get())
            except ValueError as exc:
                raise ValueError("Depth offset must be a number.") from exc
            return {
                "mode": "profile",
                "input_dir": input_dir,
                "run": run,
                "output_dir": output_dir,
                "depth_offset": depth_offset,
            }

        try:
            trim_percent = float(self.trim_percent.get())
        except ValueError as exc:
            raise ValueError("Trim percent must be a number.") from exc
        if trim_percent < 0 or trim_percent >= 50:
            raise ValueError("Trim percent must be between 0 and 49.")
        return {
            "mode": "lab",
            "input_dir": input_dir,
            "run": run,
            "output_dir": output_dir,
            "trim_percent": trim_percent,
        }

    def _run_pipeline(self, params: dict[str, object]) -> None:
        try:
            if params["mode"] == "profile":
                result = run_profile(
                    input_dir=params["input_dir"],
                    run=params["run"],
                    output_dir=params["output_dir"],
                    depth_offset_m=params["depth_offset"],
                )
            else:
                result = run_lab(
                    input_dir=params["input_dir"],
                    run=params["run"],
                    output_dir=params["output_dir"],
                    trim_percent=params["trim_percent"],
                )
            self._events.put(("done", result))
        except Exception:
            self._events.put(("error", traceback.format_exc()))

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self._events.get_nowait()
                if kind == "done":
                    self._handle_done(payload)
                elif kind == "error":
                    self._handle_error(str(payload))
        except queue.Empty:
            pass
        self.after(120, self._drain_events)

    def _handle_done(self, result: dict[str, object]) -> None:
        self._last_output_dir = Path(result["output_dir"])
        self._last_dashboard = Path(result["dashboard"])

        self._append_log(f"Output folder: {self._last_output_dir}")
        self._append_log(f"Dashboard: {self._last_dashboard}")
        self._append_log(f"CSV files created: {len(result['csv_files'])}")
        self._append_log(f"PNG plots created: {len(result['plot_files'])}")
        for error in result["errors"]:
            self._append_log(f"Warning: {error}")

        self.status.set("Done")
        self.process_button.state(["!disabled"])
        self.open_folder_button.state(["!disabled"])
        self.open_dashboard_button.state(["!disabled"])
        messagebox.showinfo(APP_TITLE, "Processing finished successfully.")

    def _handle_error(self, text: str) -> None:
        self._append_log(text)
        self.status.set("Failed")
        self.process_button.state(["!disabled"])
        messagebox.showerror(APP_TITLE, "Processing failed. See the log for details.")

    def _append_log(self, text: str) -> None:
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def _open_output_folder(self) -> None:
        if self._last_output_dir and self._last_output_dir.exists():
            os.startfile(self._last_output_dir)

    def _open_dashboard(self) -> None:
        if self._last_dashboard and self._last_dashboard.exists():
            webbrowser.open(self._last_dashboard.resolve().as_uri())


def _run_self_test(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Hidden packaged-app smoke test.")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--mode", choices=["profile", "lab"], default="profile")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--depth-offset-m", type=float, default=0.0)
    parser.add_argument("--trim-percent", type=float, default=30.0)
    args = parser.parse_args(argv)

    if args.mode == "profile":
        result = run_profile(args.input_dir, args.run, args.output_dir, args.depth_offset_m)
    else:
        result = run_lab(args.input_dir, args.run, args.output_dir, args.trim_percent)

    report = {
        "output_dir": str(result["output_dir"]),
        "dashboard": str(result["dashboard"]),
        "csv_files": [str(path) for path in result["csv_files"]],
        "plot_files": [str(path) for path in result["plot_files"]],
        "errors": result["errors"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if not result["errors"] else 2


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv:
        return _run_self_test(argv)

    app = IOPProbeApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
