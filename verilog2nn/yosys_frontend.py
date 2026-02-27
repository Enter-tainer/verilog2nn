"""Yosys synthesis frontend: read Verilog, synthesize to gate-level, output JSON."""

import json
import os
import subprocess
import tempfile
from pathlib import Path


YOSYS_BIN = os.environ.get("YOSYS_BIN", "yosys")


def _quote_yosys_arg(value: str | Path) -> str:
    """Quote a command argument for a Yosys script file."""
    s = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def find_yosys() -> str:
    """Find yosys binary, checking micromamba env if default not found."""
    # Check if default yosys is available
    try:
        subprocess.run(
            [YOSYS_BIN, "-V"],
            capture_output=True,
            check=True,
        )
        return YOSYS_BIN
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Try micromamba env
    mamba_root = os.environ.get("MAMBA_ROOT_PREFIX", "/tmp/mamba-root")
    mamba_yosys = os.path.join(mamba_root, "envs", "eda", "bin", "yosys")
    if os.path.isfile(mamba_yosys):
        return mamba_yosys

    raise FileNotFoundError(
        "yosys not found. Install via conda-forge or set YOSYS_BIN env var."
    )


def synthesize(verilog_path: Path, top: str | None = None) -> dict:
    """Synthesize Verilog to gate-level JSON netlist using Yosys.

    Args:
        verilog_path: Path to input Verilog file.
        top: Top module name. If None, Yosys auto-detects.

    Returns:
        Parsed JSON netlist dict.
    """
    yosys = find_yosys()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        json_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ys", delete=False) as f:
        script_path = f.name

    try:
        script_lines = [f"read_verilog {_quote_yosys_arg(verilog_path)}"]
        if top:
            script_lines.append(f"synth -top {top} -flatten")
        else:
            script_lines.append("synth -flatten")
        script_lines.extend(
            [
                "abc -g AND,OR,XOR",
                "clean -purge",
                f"write_json {_quote_yosys_arg(json_path)}",
            ]
        )
        with open(script_path, "w", encoding="utf-8") as f:
            f.write("\n".join(script_lines))

        result = subprocess.run(
            [yosys, script_path],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Yosys synthesis failed (exit code {result.returncode}):\n"
                f"{result.stderr}"
            )

        with open(json_path) as f:
            netlist = json.load(f)

        return netlist
    finally:
        if os.path.exists(json_path):
            os.unlink(json_path)
        if os.path.exists(script_path):
            os.unlink(script_path)
