#!/usr/bin/env python3
"""Test runner for verilog2nn - runs all tests in order by phase."""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch

# Ensure verilog2nn is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from verilog2nn.yosys_frontend import synthesize
from verilog2nn.netlist_parser import parse_netlist
from verilog2nn.topo_sort import topological_sort_layers
from verilog2nn.gate_mapper import compile_to_nn
from verilog2nn.verify import verify

VERILOG_DIR = Path(__file__).parent / "verilog"

# Large crypto circuits - skip in T8/T9 bulk scans (tested individually in T12-T14)
SKIP_BULK = {"md5_full.v", "sha256.v", "aes128.v"}


def test_phase2_yosys_json(verbose=False):
    """T8: Verify Yosys JSON parsing."""
    print("\n=== T8: Yosys JSON Parsing ===")
    for vf in sorted(VERILOG_DIR.glob("*.v")):
        if vf.name in SKIP_BULK:
            continue
        netlist = synthesize(vf)
        modules = netlist.get("modules", {})
        assert modules, f"No modules in {vf.name}"
        for mod_name, mod in modules.items():
            ports = mod.get("ports", {})
            cells = mod.get("cells", {})
            assert ports, f"No ports in {vf.name}:{mod_name}"
            if verbose:
                print(
                    f"  {vf.name}: module={mod_name}, "
                    f"ports={len(ports)}, cells={len(cells)}"
                )
    print("  PASSED: All Verilog files produce valid JSON netlist")
    return True


def test_phase3_topo_sort(verbose=False):
    """T9: Verify topological sort and layer assignment."""
    print("\n=== T9: Topological Sort ===")
    for vf in sorted(VERILOG_DIR.glob("*.v")):
        if vf.name == "md5_round.v" or vf.name in SKIP_BULK:
            continue  # skip complex circuits
        netlist = synthesize(vf)
        circuit = parse_netlist(netlist)
        layers = topological_sort_layers(circuit)

        # Verify: gates in earlier layers don't depend on later layers
        net_available = set(circuit.input_bits) | set(circuit.const_nets.keys())
        for li, layer in enumerate(layers):
            for gate in layer:
                for inp in gate.input_nets:
                    assert inp in net_available, (
                        f"Gate {gate.name} in layer {li} depends on "
                        f"unavailable net {inp}"
                    )
            for gate in layer:
                net_available.add(gate.output_net)

        if verbose:
            print(
                f"  {vf.name}: {len(circuit.gates)} gates, "
                f"{len(layers)} layers, "
                f"inputs={len(circuit.input_bits)}, "
                f"outputs={len(circuit.output_bits)}"
            )
    print("  PASSED: All circuits have valid topological ordering")
    return True


def test_phase4_safetensors(verbose=False):
    """T10: Verify safetensors output."""
    print("\n=== T10: safetensors Output ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        # Test with a simple circuit
        vf = VERILOG_DIR / "and_gate.v"
        netlist = synthesize(vf)
        circuit = parse_netlist(netlist)
        layers = topological_sort_layers(circuit)
        compile_to_nn(circuit, layers, output_dir)

        # Check files exist
        assert (output_dir / "weights.safetensors").exists()
        assert (output_dir / "model.py").exists()
        assert (output_dir / "sparse_linear.py").exists()
        assert (output_dir / "inference.py").exists()

        # Load and verify weights
        from safetensors.torch import load_file
        weights = load_file(str(output_dir / "weights.safetensors"))
        assert "network_meta" in weights
        assert "output_indices" in weights
        meta = weights["network_meta"]
        assert meta[0].item() == len(circuit.input_bits)
        assert meta[1].item() == len(circuit.output_bits)

        # Load model and do a basic forward pass
        sys.path.insert(0, str(output_dir))
        # Need to reload model modules since we might have loaded them before
        for mod_name in ["model", "sparse_linear"]:
            if mod_name in sys.modules:
                del sys.modules[mod_name]
        from model import VerilogNN
        model = VerilogNN(str(output_dir / "weights.safetensors"))
        model.eval()
        x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (4, 1), f"Expected shape (4, 1), got {y.shape}"
        if verbose:
            print(f"  AND gate outputs: {y.squeeze().tolist()}")

    print("  PASSED: safetensors file loads correctly, shapes match")
    return True


def test_verification(name, vf_name, exhaustive=True, num_tests=100, verbose=False):
    """Run verification for a single Verilog file."""
    print(f"\n=== {name} ===")
    vf = VERILOG_DIR / vf_name
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        netlist = synthesize(vf)
        circuit = parse_netlist(netlist)
        layers = topological_sort_layers(circuit)
        compile_to_nn(circuit, layers, output_dir)

        ok = verify(
            vf,
            output_dir,
            num_tests=num_tests,
            exhaustive=exhaustive,
        )
        if ok:
            print(f"  PASSED")
        else:
            print(f"  FAILED")
        return ok


def main():
    quick = "--quick" in sys.argv
    results = {}

    # Phase 2: Yosys frontend
    print("\n" + "=" * 60)
    print("PHASE 2: Yosys Frontend")
    print("=" * 60)
    results["T8"] = test_phase2_yosys_json(verbose=True)

    # Phase 3: Topo sort
    print("\n" + "=" * 60)
    print("PHASE 3: Netlist Parsing & Topo Sort")
    print("=" * 60)
    results["T9"] = test_phase3_topo_sort(verbose=True)

    # Phase 4: Gate mapping + safetensors
    print("\n" + "=" * 60)
    print("PHASE 4: Gate-to-ReLU Mapping")
    print("=" * 60)
    results["T10"] = test_phase4_safetensors(verbose=True)

    # Phase 5: Basic verification
    print("\n" + "=" * 60)
    print("PHASE 5: Basic Verification (T1-T3)")
    print("=" * 60)
    results["T1"] = test_verification("T1: NOT Gate", "not_gate.v")
    results["T2a"] = test_verification("T2a: AND Gate", "and_gate.v")
    results["T2b"] = test_verification("T2b: OR Gate", "or_gate.v")
    results["T2c"] = test_verification("T2c: XOR Gate", "xor_gate.v")
    results["T3"] = test_verification("T3: Combo Expression", "combo_expr.v")

    # Phase 6: Multi-bit and complex
    print("\n" + "=" * 60)
    print("PHASE 6: Multi-bit & Complex Circuits (T4-T7)")
    print("=" * 60)
    results["T4"] = test_verification("T4: 4-bit Adder", "adder4.v")
    results["T5"] = test_verification(
        "T5: 8-bit Comparator", "cmp8.v", exhaustive=False, num_tests=1000
    )
    results["T6"] = test_verification("T6: 4-to-1 MUX", "mux4.v")
    results["T7"] = test_verification(
        "T7: Module Instantiation (2-bit adder)", "multi_module.v"
    )

    # Phase 7: Stress test
    print("\n" + "=" * 60)
    print("PHASE 7: End-to-End Stress Test (T11)")
    print("=" * 60)
    results["T11"] = test_verification(
        "T11: MD5 Round", "md5_round.v", exhaustive=False, num_tests=100
    )

    # Phase 8: Cryptographic algorithms (slow, skip with --quick)
    if not quick:
        print("\n" + "=" * 60)
        print("PHASE 8: Cryptographic Algorithms (T12-T14)")
        print("=" * 60)
        results["T12"] = test_verification(
            "T12: Full MD5 (64 steps)", "md5_full.v",
            exhaustive=False, num_tests=20,
        )
        results["T13"] = test_verification(
            "T13: SHA-256 (64 rounds)", "sha256.v",
            exhaustive=False, num_tests=10,
        )
        results["T14"] = test_verification(
            "T14: AES-128 (10 rounds)", "aes128.v",
            exhaustive=False, num_tests=10,
        )
    else:
        print("\n" + "=" * 60)
        print("PHASE 8: Skipped (--quick mode)")
        print("=" * 60)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for test_id, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {test_id}: {status}")

    if all_pass:
        print("\nALL TESTS PASSED!")
    else:
        print("\nSOME TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
