from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _run_command(*args: str) -> int:
    completed = subprocess.run([sys.executable, *args], cwd=ROOT)
    return completed.returncode


def main() -> int:
    erase_exit = _run_command("-m", "coverage", "erase")
    if erase_exit != 0:
        return erase_exit

    test_exit = _run_command("-m", "coverage", "run", "-m", "unittest", "discover", "-s", "tests", "-v")
    report_exit = _run_command("-m", "coverage", "report")

    if test_exit != 0:
        return test_exit
    return report_exit


if __name__ == "__main__":
    raise SystemExit(main())