"""CLI entry point for verilog2nn compiler."""

import argparse
import sys
from pathlib import Path

from verilog2nn.yosys_frontend import synthesize
from verilog2nn.netlist_parser import parse_netlist
from verilog2nn.topo_sort import topological_sort_layers
from verilog2nn.gate_mapper import compile_to_nn
from verilog2nn.verify import verify


def main():
    parser = argparse.ArgumentParser(
        description="Compile Verilog combinational logic to PyTorch neural network"
    )
    parser.add_argument("input", help="Input Verilog file (.v)")
    parser.add_argument(
        "-o", "--output", default="output", help="Output directory (default: output)"
    )
    parser.add_argument(
        "--top", default=None, help="Top module name (auto-detect if not specified)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify NN output against iverilog simulation",
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=100,
        help="Number of random test vectors for verification (default: 100)",
    )
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="Use exhaustive testing (all input combinations)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Yosys synthesis
    print(f"[1/4] Synthesizing {input_path} with Yosys...")
    netlist_json = synthesize(input_path, top=args.top)

    # Step 2: Parse netlist
    print("[2/4] Parsing netlist and building DAG...")
    circuit = parse_netlist(netlist_json, top=args.top)

    # Step 3: Topological sort
    print("[3/4] Topological sorting and layer assignment...")
    layers = topological_sort_layers(circuit)

    # Step 4: Compile to NN
    print("[4/4] Compiling to PyTorch neural network...")
    compile_to_nn(circuit, layers, output_dir)

    print(f"\nOutput files in {output_dir}/:")
    print(f"  model.py       - PyTorch nn.Module definition")
    print(f"  weights.safetensors - Model weights")
    print(f"  inference.py   - Standalone inference script")

    # Optional verification
    if args.verify:
        print("\n[verify] Running verification...")
        ok = verify(
            input_path,
            output_dir,
            num_tests=args.num_tests,
            exhaustive=args.exhaustive,
            top=args.top,
        )
        if not ok:
            print("VERIFICATION FAILED", file=sys.stderr)
            sys.exit(1)
        print("VERIFICATION PASSED")


if __name__ == "__main__":
    main()
