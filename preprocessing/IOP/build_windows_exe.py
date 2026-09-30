from __future__ import annotations

import shutil
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
APP_NAME = "IOPProbeProcessor"


def main() -> int:
    for local_deps in [PROJECT_DIR / ".build_deps", PROJECT_DIR / ".codex_mpl"]:
        if local_deps.exists():
            sys.path.insert(0, str(local_deps))

    try:
        from PyInstaller.__main__ import run as pyinstaller_run
    except ImportError as exc:
        raise SystemExit(
            "PyInstaller is not installed. Install it with:\n"
            "  python -m pip install pyinstaller\n"
            "or into the local build folder with:\n"
            "  python -m pip install --target .build_deps pyinstaller"
        ) from exc

    for folder in [PROJECT_DIR / "build", PROJECT_DIR / "dist" / APP_NAME]:
        if folder.exists():
            shutil.rmtree(folder)

    args = [
        str(PROJECT_DIR / "iop_probe_app.py"),
        "--name",
        APP_NAME,
        "--windowed",
        "--onedir",
        "--clean",
        "--noconfirm",
        "--paths",
        str(PROJECT_DIR),
        "--collect-all",
        "matplotlib",
        "--exclude-module",
        "pandas.tests",
        "--exclude-module",
        "matplotlib.tests",
        "--exclude-module",
        "pytest",
        "--hidden-import",
        "matplotlib.backends.backend_agg",
        "--hidden-import",
        "PIL._tkinter_finder",
        "--hidden-import",
        "tkinter",
    ]
    if (PROJECT_DIR / ".codex_mpl").exists():
        args.extend(["--paths", str(PROJECT_DIR / ".codex_mpl")])

    pyinstaller_run(args)
    guide = PROJECT_DIR / "WINDOWS_APP_USER_GUIDE.md"
    if guide.exists():
        shutil.copy2(guide, PROJECT_DIR / "dist" / APP_NAME / "README_FOR_USERS.md")
    print(f"\nBuilt: {PROJECT_DIR / 'dist' / APP_NAME / (APP_NAME + '.exe')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
