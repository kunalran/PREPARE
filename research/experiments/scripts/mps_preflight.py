from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

from prepare_experiments import ensure_venv  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]


def query_sw_vers() -> dict[str, str]:
    try:
        proc = subprocess.run(
            ["sw_vers"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return {"error": str(exc)}

    if proc.returncode != 0:
        return {"error": proc.stderr.strip() or proc.stdout.strip() or "sw_vers failed"}

    info: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        info[key.strip()] = value.strip()
    return info


def collect_device_status() -> dict[str, object]:
    status: dict[str, object] = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "system": platform.system(),
        "torch_version": torch.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "sw_vers": query_sw_vers() if sys.platform == "darwin" else {},
    }

    status["cpu_selected"] = True
    status["selected_device"] = "cpu"
    status["mps_reason"] = ""

    if sys.platform != "darwin":
        status["mps_reason"] = "MPS is only available on macOS."
        return status

    if not torch.backends.mps.is_built():
        status["mps_reason"] = "This Torch build was compiled without MPS support."
        return status

    if torch.backends.mps.is_available():
        status["cpu_selected"] = False
        status["selected_device"] = "mps"
        status["mps_reason"] = "MPS is available."
        return status

    try:
        torch.ones(1, device="mps")
    except Exception as exc:  # pragma: no cover - runtime-dependent
        status["mps_reason"] = str(exc)
        product_version = str(status.get("sw_vers", {}).get("ProductVersion", ""))
        if "MacOS 13.0+" in status["mps_reason"] and product_version:
            major = product_version.split(".", 1)[0]
            if major.isdigit() and int(major) >= 13:
                status["mps_reason_detail"] = (
                    "Torch reports an OS-version failure even though sw_vers shows a supported "
                    "macOS version. This points to a local Torch/runtime detection mismatch."
                )
    else:  # pragma: no cover - defensive only
        status["cpu_selected"] = False
        status["selected_device"] = "mps"
        status["mps_reason"] = "MPS tensor allocation succeeded after the availability probe."
    return status


def select_torch_device(requested_device: str) -> str:
    status = collect_device_status()
    if requested_device == "cpu":
        return "cpu"
    if requested_device == "mps":
        if status["selected_device"] != "mps":
            raise RuntimeError(f"MPS requested but unavailable: {status['mps_reason']}")
        return "mps"
    return str(status["selected_device"])


def write_device_status(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = collect_device_status()
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Torch MPS availability for the newtests venv.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "newtests" / "focused_1d_15d_expansion" / "results" / "environment_status" / "device_status.json",
    )
    return parser.parse_args()


def main() -> None:
    ensure_venv()
    args = parse_args()
    output = write_device_status(args.output_path)
    print(output)


if __name__ == "__main__":
    main()
