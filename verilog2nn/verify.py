"""Verification: compare PyTorch NN output against iverilog simulation."""

import importlib
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from verilog2nn.yosys_frontend import synthesize
from verilog2nn.netlist_parser import parse_netlist


IVERILOG_BIN = os.environ.get("IVERILOG_BIN", "iverilog")
VVP_BIN = os.environ.get("VVP_BIN", "vvp")


def find_iverilog() -> tuple[str, str]:
    """Find iverilog and vvp binaries."""
    for iverilog, vvp in [
        (IVERILOG_BIN, VVP_BIN),
        ("/tmp/mamba-root/envs/eda/bin/iverilog", "/tmp/mamba-root/envs/eda/bin/vvp"),
    ]:
        try:
            subprocess.run([iverilog, "-V"], capture_output=True, check=True)
            return iverilog, vvp
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    raise FileNotFoundError(
        "iverilog not found. Install via conda-forge or set IVERILOG_BIN env var."
    )


def _get_port_widths_from_netlist(
    netlist_json: dict, top: str | None = None
) -> tuple[str, dict[str, int], dict[str, int]]:
    """Extract port widths from Yosys JSON netlist.

    Returns:
        (module_name, input_widths, output_widths)
    """
    modules = netlist_json.get("modules", {})
    if not modules:
        raise ValueError("No modules found in netlist JSON")

    if top:
        if top not in modules:
            raise ValueError(f"Module '{top}' not found. Available: {list(modules)}")
        module_name = top
        module = modules[top]
    else:
        for name, mod in modules.items():
            attrs = mod.get("attributes", {})
            if attrs.get("top") == "00000000000000000000000000000001":
                module_name = name
                module = mod
                break
        else:
            module_name = next(iter(modules))
            module = modules[module_name]

    ports = module.get("ports", {})
    input_widths = {}
    output_widths = {}

    for port_name, port_info in ports.items():
        direction = port_info["direction"]
        bits = port_info["bits"]
        width = len(bits)
        if direction == "input":
            input_widths[port_name] = width
        elif direction == "output":
            output_widths[port_name] = width

    return module_name, input_widths, output_widths


def _find_top_module_in_verilog(verilog_path: Path, target: str) -> str:
    """Find the top module name in a Verilog file for iverilog.

    For multi-module files, iverilog needs the top module name to instantiate.
    We use `target` which comes from the Yosys netlist (the correct top module).
    """
    return target


def _generate_testbench(
    input_widths: dict[str, int],
    output_widths: dict[str, int],
    test_vectors: list[dict[str, int]],
    top_module: str,
) -> str:
    """Generate a Verilog testbench for the given test vectors."""
    lines = ["`timescale 1ns/1ps", "", "module tb;", ""]

    # Declare signals
    for name, width in input_widths.items():
        if width > 1:
            lines.append(f"  reg [{width-1}:0] {name};")
        else:
            lines.append(f"  reg {name};")

    for name, width in output_widths.items():
        if width > 1:
            lines.append(f"  wire [{width-1}:0] {name};")
        else:
            lines.append(f"  wire {name};")

    lines.append("")

    # Instantiate DUT
    ports = []
    for name in list(input_widths) + list(output_widths):
        ports.append(f".{name}({name})")
    lines.append(f"  {top_module} dut ({', '.join(ports)});")
    lines.append("")

    # Test stimulus
    lines.append("  initial begin")
    for vec in test_vectors:
        for name, val in vec.items():
            if name in input_widths:
                lines.append(f"    {name} = {input_widths[name]}'d{val};")
        lines.append("    #10;")
        # Display outputs
        fmt_parts = []
        val_parts = []
        for name in output_widths:
            fmt_parts.append(f"{name}=%0b")
            val_parts.append(name)
        for name in input_widths:
            fmt_parts.append(f"{name}=%0b")
            val_parts.append(name)
        fmt = " ".join(fmt_parts)
        vals = ", ".join(val_parts)
        lines.append(f'    $display("{fmt}", {vals});')
    lines.append("    $finish;")
    lines.append("  end")
    lines.append("")
    lines.append("endmodule")
    return "\n".join(lines)


def _run_iverilog(
    verilog_path: Path,
    testbench: str,
    iverilog_bin: str,
    vvp_bin: str,
    top_module: str,
) -> str:
    """Run iverilog simulation and return output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tb_path = Path(tmpdir) / "tb.v"
        tb_path.write_text(testbench)
        out_path = Path(tmpdir) / "sim.vvp"

        # Compile (use -s to specify top for multi-module files)
        result = subprocess.run(
            [iverilog_bin, "-o", str(out_path), "-s", "tb",
             str(verilog_path), str(tb_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"iverilog compilation failed:\n{result.stderr}")

        # Run
        result = subprocess.run(
            [vvp_bin, str(out_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"vvp simulation failed:\n{result.stderr}")

        return result.stdout


def _generate_test_vectors(
    input_widths: dict[str, int],
    num_tests: int,
    exhaustive: bool,
) -> list[dict[str, int]]:
    """Generate test input vectors."""
    total_bits = sum(input_widths.values())

    if exhaustive or (2**total_bits <= num_tests):
        # Exhaustive
        vectors = []
        for val in range(2**total_bits):
            vec = {}
            offset = 0
            for name, width in input_widths.items():
                mask = (1 << width) - 1
                vec[name] = (val >> offset) & mask
                offset += width
            vectors.append(vec)
        return vectors
    else:
        # Random
        vectors = []
        for _ in range(num_tests):
            vec = {}
            for name, width in input_widths.items():
                vec[name] = random.randint(0, (1 << width) - 1)
            vectors.append(vec)
        return vectors


def _parse_simulation_output(
    output: str,
    output_widths: dict[str, int],
    input_widths: dict[str, int],
) -> list[dict]:
    """Parse iverilog $display output into list of result dicts."""
    results = []
    for line in output.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        result = {}
        for part in line.split():
            m = re.match(r"(\w+)=(\d+)", part)
            if m:
                name, val_str = m.groups()
                result[name] = int(val_str, 2)  # binary
        if result:
            results.append(result)
    return results


def _circuit_input_from_vector(
    vec: dict[str, int],
    input_names: list[str],
) -> list[int]:
    """Convert a test vector dict to flat bit list matching circuit input order.

    input_names comes from the circuit parser and has names like "a", "a[0]", "a[1]"
    for multi-bit signals.
    """
    bits = []
    for name in input_names:
        m = re.match(r"(\w+)\[(\d+)\]", name)
        if m:
            port_name = m.group(1)
            bit_idx = int(m.group(2))
            val = (vec[port_name] >> bit_idx) & 1
        else:
            # Single bit signal
            if name in vec:
                val = vec[name] & 1
            else:
                val = 0
        bits.append(val)
    return bits


def _circuit_output_to_dict(
    output_bits: list[int],
    output_names: list[str],
) -> dict[str, int]:
    """Convert flat output bit list to dict of port values."""
    result = {}
    for name, bit_val in zip(output_names, output_bits):
        m = re.match(r"(\w+)\[(\d+)\]", name)
        if m:
            port_name = m.group(1)
            bit_idx = int(m.group(2))
            result.setdefault(port_name, 0)
            result[port_name] |= (int(bit_val) & 1) << bit_idx
        else:
            result[name] = int(bit_val) & 1
    return result


def verify(
    verilog_path: Path,
    output_dir: Path,
    num_tests: int = 100,
    exhaustive: bool = False,
    top: str | None = None,
) -> bool:
    """Verify compiled NN against iverilog simulation.

    Returns True if all tests pass, False otherwise.
    """
    iverilog_bin, vvp_bin = find_iverilog()

    # Synthesize to get accurate port info from the actual top module
    netlist_json = synthesize(verilog_path, top=top)
    circuit = parse_netlist(netlist_json, top=top)
    module_name, input_widths, output_widths = _get_port_widths_from_netlist(
        netlist_json, top=top
    )

    # Generate test vectors
    vectors = _generate_test_vectors(input_widths, num_tests, exhaustive)
    print(f"  Testing with {len(vectors)} vectors...")

    # Run iverilog simulation
    testbench = _generate_testbench(
        input_widths, output_widths, vectors, module_name,
    )
    sim_output = _run_iverilog(
        verilog_path, testbench, iverilog_bin, vvp_bin, module_name
    )
    sim_results = _parse_simulation_output(sim_output, output_widths, input_widths)

    if len(sim_results) != len(vectors):
        print(f"  ERROR: Expected {len(vectors)} results, got {len(sim_results)}")
        return False

    # Load NN model (force reload to handle multiple calls in same process)
    sys.path.insert(0, str(output_dir))
    if "model" in sys.modules:
        del sys.modules["model"]
    import model as model_mod
    importlib.reload(model_mod)
    model = model_mod.VerilogNN(str(output_dir / "weights.safetensors"))
    model.eval()

    # Compare
    mismatches = 0
    for i, (vec, sim_res) in enumerate(zip(vectors, sim_results)):
        # Build NN input
        nn_input = _circuit_input_from_vector(vec, circuit.input_names)
        x = torch.tensor([nn_input], dtype=torch.float64)
        with torch.no_grad():
            y = model(x)
        nn_output = y[0].tolist()

        # Convert NN output to dict
        nn_dict = _circuit_output_to_dict(nn_output, circuit.output_names)

        # Compare with simulation
        for port_name in output_widths:
            sim_val = sim_res.get(port_name, 0)
            nn_val = nn_dict.get(port_name, 0)
            if sim_val != nn_val:
                if mismatches < 10:
                    print(
                        f"  MISMATCH at vector {i}: {vec} -> "
                        f"{port_name}: sim={sim_val}, nn={nn_val}"
                    )
                mismatches += 1

    if mismatches > 0:
        print(f"  {mismatches} mismatches out of {len(vectors)} tests")
        return False

    print(f"  All {len(vectors)} tests passed!")
    return True
