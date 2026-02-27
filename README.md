# verilog2nn

Compile Verilog combinational logic into equivalent PyTorch neural networks. No training, no approximation — the network computes the exact same function as the original circuit, for all inputs.

## Why?

Jane Street published a [puzzle](https://blog.janestreet.com/can-you-reverse-engineer-our-neural-network/) where they hand-built a neural network that computed MD5, then challenged people to reverse-engineer it. The solver found that the network implemented logic gates via ReLU layers — AND, OR, XOR all have clean representations as small linear+ReLU combos.

That got me thinking: if you can hand-build a circuit-as-NN, why not automate it? Take any Verilog module, synthesize it down to gates, and mechanically map each gate to its ReLU equivalent. The result is a neural network that is *provably* equivalent to the original circuit — not trained to approximate it, but constructed to replicate it exactly.

## How it works

```
Verilog → Yosys (synthesize to gate-level) → JSON netlist → topological sort → ReLU mapping → PyTorch model
```

1. **Yosys frontend**: reads your `.v` file, flattens hierarchy, maps everything to AND/OR/XOR/NOT via ABC
2. **Netlist parser**: builds a gate-level DAG from Yosys's JSON output
3. **Topological sort**: assigns gates to layers — gates in the same layer have no dependencies on each other
4. **Gate mapping**: each gate becomes a small linear+ReLU block:

| Gate | Formula (inputs are 0/1 floats) |
|------|------|
| NOT(x) | 1 − x |
| AND(a,b) | ReLU(a + b − 1) |
| OR(a,b) | (a + b) − ReLU(a + b − 1) |
| XOR(a,b) | (a + b) − 2·ReLU(a + b − 1) |

Gates in the same topological layer get merged into a single wide matrix multiply. Layer-to-layer connections become sparse routing matrices. The whole thing compiles down to a `nn.Module` with integer weights — no training required.

## Quick start

```bash
pip install -e .
# you also need: yosys, iverilog (for verification)

# compile a Verilog module to a PyTorch model
verilog2nn tests/verilog/adder4.v -o output/

# this produces:
#   output/model.py          — nn.Module definition
#   output/weights.safetensors — model weights
#   output/inference.py      — standalone inference script
```

## Demo: 4-bit adder

Given a simple ripple-carry adder:

```verilog
module adder4(input [3:0] a, input [3:0] b, output [4:0] sum);
  assign sum = a + b;
endmodule
```

verilog2nn synthesizes it into 17 gates across 7 layers, producing a network that maps 8 input bits → 5 output bits. Exhaustive verification over all 256 input combinations confirms exact equivalence:

```
$ python tests/run_tests.py
...
=== T4: 4-bit Adder ===
  Verified 256/256 inputs: ALL MATCH
  PASSED
```

It also handles more serious circuits — the test suite includes an MD5 round function (797 gates), verified against iverilog simulation.

## Tests

```bash
python tests/run_tests.py
```

Covers single gates (NOT, AND, OR, XOR), combinational expressions, 4-bit adder, 8-bit comparator, 4-to-1 MUX, multi-module instantiation, and an MD5 round function as a stress test.

## Acknowledgments

Inspired by Jane Street's ["Can you reverse engineer our neural network?"](https://blog.janestreet.com/can-you-reverse-engineer-our-neural-network/) puzzle, which demonstrated that deterministic computation (including MD5!) can be faithfully encoded in ReLU networks.
