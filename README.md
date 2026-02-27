# verilog2nn

Compile Verilog combinational logic into equivalent PyTorch neural networks. No training, no approximation — the network computes the exact same function as the original circuit, for all inputs.

## Why?

Jane Street published a [puzzle](https://blog.janestreet.com/can-you-reverse-engineer-our-neural-network/) where they hand-built a neural network that computed MD5, then challenged people to reverse-engineer it. The solver found that the network implemented logic gates via ReLU layers — AND, OR, XOR all have clean representations as small linear+ReLU combos.

That got me thinking: if you can hand-build a circuit-as-NN, why not automate it? Take any Verilog module, synthesize it down to gates, and mechanically map each gate to its ReLU equivalent. The result is a neural network that is *provably* equivalent to the original circuit — not trained to approximate it, but constructed to replicate it exactly.

## How it works

```
Verilog → Yosys (synthesize to AND+NOT) → JSON netlist → topological sort → SparseLinear+ReLU MLP → PyTorch model
```

1. **Yosys frontend**: reads your `.v` file, flattens hierarchy, maps everything to AND+NOT gates via ABC (`abc -g AND`). AND+NOT is functionally complete — any Boolean function can be expressed with just these two gates.
2. **Netlist parser**: builds a gate-level DAG from Yosys's JSON output
3. **Topological sort**: assigns gates to layers — gates in the same layer have no dependencies on each other
4. **Gate mapping**: AND gates become ReLU neurons, NOT gates are folded into weights:

| Gate | Formula | Neural network equivalent |
|------|---------|--------------------------|
| AND(a,b) | ReLU(a + b − 1) | SparseLinear (weights: +1, +1, bias: −1) → ReLU |
| AND(a, NOT(b)) | ReLU(a − b) | SparseLinear (weights: +1, −1, bias: 0) → ReLU |
| AND(NOT(a), NOT(b)) | ReLU(−a − b + 1) | SparseLinear (weights: −1, −1, bias: +1) → ReLU |

NOT gates don't produce separate neurons — they're absorbed into the SparseLinear weights of downstream AND gates. This means every layer is just **SparseLinear → ReLU**, a clean MLP architecture.

The generated model consists of:
- `sparse_linear.py` — `SparseLinear(nn.Module)` using COO format weights
- `model.py` — `VerilogNN(nn.Module)` stacking SparseLinear + ReLU layers
- `weights.safetensors` — sparse weights in COO format (indices + values + bias per layer)

## Quick start

```bash
pip install -e .
# you also need: yosys, iverilog (for verification)

# compile a Verilog module to a PyTorch model
verilog2nn tests/verilog/adder4.v -o output/

# this produces:
#   output/sparse_linear.py    — SparseLinear module
#   output/model.py            — nn.Module definition
#   output/weights.safetensors — model weights (COO sparse)
#   output/inference.py        — standalone inference script
```

## Demo: 4-bit adder

Given a simple ripple-carry adder:

```verilog
module adder4(input [3:0] a, input [3:0] b, output [4:0] sum);
  assign sum = a + b;
endmodule
```

verilog2nn synthesizes it down to AND+NOT gates, folds NOTs into weights, and produces a pure SparseLinear → ReLU MLP that maps 8 input bits → 5 output bits. Exhaustive verification over all 256 input combinations confirms exact equivalence:

```
$ python tests/run_tests.py
...
=== T4: 4-bit Adder ===
  Verified 256/256 inputs: ALL MATCH
  PASSED
```

The test suite also covers crypto-scale circuits: full MD5 (~52K gates), SHA-256 (~136K gates), and AES-128 (~171K gates), all verified bit-exact against iverilog simulation.

## Tests

```bash
python tests/run_tests.py          # full test suite (includes crypto circuits)
python tests/run_tests.py --quick  # skip crypto tests (faster, used in CI)
```

Test coverage:
- **T1–T3**: Single gates (NOT, AND, OR/XOR via AND+NOT decomposition)
- **T4**: 4-bit adder
- **T5**: Combinational expressions
- **T6**: 8-bit comparator
- **T7**: 4-to-1 MUX
- **T8**: Multi-module instantiation
- **T9**: 8-bit ripple-carry adder
- **T10**: Generated model structure validation
- **T11**: MD5 round function (797 gates)
- **T12**: Full MD5 (~52K gates) — `--quick` skips
- **T13**: SHA-256 (~136K gates) — `--quick` skips
- **T14**: AES-128 (~171K gates) — `--quick` skips

## Acknowledgments

Inspired by Jane Street's ["Can you reverse engineer our neural network?"](https://blog.janestreet.com/can-you-reverse-engineer-our-neural-network/) puzzle, which demonstrated that deterministic computation (including MD5!) can be faithfully encoded in ReLU networks.
