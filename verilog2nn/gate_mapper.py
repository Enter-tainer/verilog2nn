"""Map gate-level circuits to PyTorch neural networks (SparseLinear + ReLU MLP).

Only AND and NOT gates (from abc -g AND). NOT gates are folded into weights:
    NOT(x) = 1 - x  →  weight=-1, bias_contribution=+1
    AND(a,b) = ReLU(a + b - 1)

If an AND input comes through a NOT gate, the weight flips to -1 and the
bias adjusts from -1 to 0 (since NOT(x) contributes 1-x instead of x).

Each topological layer of AND gates becomes: SparseLinear → ReLU
State vector grows by appending each layer's outputs.

NOT outputs that are circuit outputs get a dedicated affine output layer.
"""

import textwrap
from pathlib import Path

import torch
from safetensors.torch import save_file

from verilog2nn.netlist_parser import Circuit, Gate


def compile_to_nn(
    circuit: Circuit,
    layers: list[list[Gate]],
    output_dir: Path,
) -> None:
    """Compile circuit to PyTorch NN with SparseLinear layers."""
    num_inputs = len(circuit.input_bits)
    num_outputs = len(circuit.output_bits)

    # Assign indices: net_id -> index in current state vector
    net_to_idx: dict[int, int] = {}
    for i, bit in enumerate(circuit.input_bits):
        net_to_idx[bit] = i

    # Also assign indices for constant nets
    current_size = num_inputs
    for net_id in circuit.const_nets:
        net_to_idx[net_id] = current_size
        current_size += 1

    # Constant values vector (appended to input)
    const_values = []
    const_indices = []
    for net_id, val in circuit.const_nets.items():
        const_values.append(float(val))
        const_indices.append(net_to_idx[net_id])

    weights = _compile_layers(
        circuit, layers, net_to_idx, current_size,
        num_inputs, num_outputs,
    )

    # Metadata tensors
    weights["output_indices"] = torch.tensor(
        weights.pop("_output_indices"), dtype=torch.int64,
    )
    weights["const_values"] = torch.tensor(
        const_values if const_values else [0.0], dtype=torch.float64,
    )
    weights["const_indices"] = torch.tensor(
        const_indices if const_indices else [0], dtype=torch.int64,
    )
    if const_values:
        weights["has_consts"] = torch.tensor([1], dtype=torch.int64)
    else:
        weights["has_consts"] = torch.tensor([0], dtype=torch.int64)

    save_file(weights, output_dir / "weights.safetensors")

    # Generate sparse_linear.py
    sparse_linear_code = _generate_sparse_linear_code()
    (output_dir / "sparse_linear.py").write_text(sparse_linear_code)

    # Generate model.py
    model_code = _generate_model_code()
    (output_dir / "model.py").write_text(model_code)

    # Generate inference.py
    inference_code = _generate_inference_code(
        num_inputs, num_outputs,
        circuit.input_names, circuit.output_names,
    )
    (output_dir / "inference.py").write_text(inference_code)


def _compile_layers(
    circuit: Circuit,
    layers: list[list[Gate]],
    net_to_idx: dict[int, int],
    current_size: int,
    num_inputs: int,
    num_outputs: int,
) -> dict:
    """Compile layers to SparseLinear COO weights with NOT folding.

    NOT gates are not emitted as neurons. Instead:
    - We track which nets are NOT outputs: not_source[not_output_net] = input_net
    - When an AND gate references a NOT output, we fold: weight=-1, bias+=1
    - NOT outputs that are circuit outputs get an affine output fixup layer
    """
    weights: dict = {}
    state_size = current_size

    # Build NOT lookup: not_output_net -> not_input_net
    not_source: dict[int, int] = {}
    for layer_gates in layers:
        for gate in layer_gates:
            if gate.gate_type == "NOT":
                not_source[gate.output_net] = gate.input_nets[0]

    # Track which NOT output nets are needed as actual state values
    # (i.e., they're circuit outputs and can't be folded)
    output_set = set(circuit.output_bits)
    not_nets_needed_in_state: set[int] = set()
    for net_id in not_source:
        if net_id in output_set:
            not_nets_needed_in_state.add(net_id)

    num_real_layers = 0

    for layer_gates in layers:
        and_gates = [g for g in layer_gates if g.gate_type == "AND"]
        if not and_gates:
            # Pure NOT layer — just record the NOT mapping, no SparseLinear
            # NOT outputs used by later AND gates are folded into those layers
            # NOT outputs that are circuit outputs will be handled by output fixup
            continue

        n_and = len(and_gates)
        layer_idx = num_real_layers
        num_real_layers += 1

        # Build COO sparse weight for this layer
        # Each AND gate: out_j = ReLU(w_a * in_a + w_b * in_b + bias)
        # Normal: w=+1 for each input, bias=-1
        # If input is NOT output: w=-1, bias contribution +1 (net: bias=0 for one NOT, +1 for two NOTs)
        row_indices = []
        col_indices = []
        values = []
        biases = []

        for j, gate in enumerate(and_gates):
            bias = -1.0
            for inp_net in gate.input_nets:
                if inp_net in not_source:
                    # This input goes through a NOT: NOT(x) = 1-x
                    # So instead of +x we have +(1-x) = -x + 1
                    actual_net = not_source[inp_net]
                    row_indices.append(j)
                    col_indices.append(net_to_idx[actual_net])
                    values.append(-1.0)
                    bias += 1.0  # compensate for the constant +1 from NOT
                else:
                    row_indices.append(j)
                    col_indices.append(net_to_idx[inp_net])
                    values.append(1.0)
            biases.append(bias)

        # Save as COO tensors
        weights[f"layer{layer_idx}.indices"] = torch.tensor(
            [row_indices, col_indices], dtype=torch.int64,
        )
        weights[f"layer{layer_idx}.values"] = torch.tensor(values, dtype=torch.float64)
        weights[f"layer{layer_idx}.bias"] = torch.tensor(biases, dtype=torch.float64)
        weights[f"layer{layer_idx}.shape"] = torch.tensor(
            [n_and, state_size], dtype=torch.int64,
        )

        # Assign output indices for AND gates
        for j, gate in enumerate(and_gates):
            net_to_idx[gate.output_net] = state_size + j

        state_size += n_and

    # Handle NOT outputs that are circuit outputs: affine fixup layer
    # out = 1 - state[source_idx], stored as sparse affine transform
    # We also handle the case where a NOT-of-NOT chain leads to an output
    if not_nets_needed_in_state:
        fixup_row = []
        fixup_col = []
        fixup_val = []
        fixup_bias = []
        fixup_j = 0
        # Map from NOT output net -> new state index
        fixup_net_to_idx: dict[int, int] = {}

        for not_net in sorted(not_nets_needed_in_state):
            # Resolve chain of NOTs to find the actual source in state
            source_net = not_source[not_net]
            inverted = True
            while source_net in not_source:
                source_net = not_source[source_net]
                inverted = not inverted

            if source_net not in net_to_idx:
                raise ValueError(
                    f"NOT source net {source_net} not found in state vector"
                )

            if inverted:
                # NOT(x) = 1 - x
                fixup_row.append(fixup_j)
                fixup_col.append(net_to_idx[source_net])
                fixup_val.append(-1.0)
                fixup_bias.append(1.0)
            else:
                # Double NOT = identity
                fixup_row.append(fixup_j)
                fixup_col.append(net_to_idx[source_net])
                fixup_val.append(1.0)
                fixup_bias.append(0.0)

            fixup_net_to_idx[not_net] = state_size + fixup_j
            fixup_j += 1

        n_fixup = fixup_j
        weights["fixup.indices"] = torch.tensor(
            [fixup_row, fixup_col], dtype=torch.int64,
        )
        weights["fixup.values"] = torch.tensor(fixup_val, dtype=torch.float64)
        weights["fixup.bias"] = torch.tensor(fixup_bias, dtype=torch.float64)
        weights["fixup.shape"] = torch.tensor(
            [n_fixup, state_size], dtype=torch.int64,
        )

        # Update net_to_idx for these NOT outputs
        for not_net, idx in fixup_net_to_idx.items():
            net_to_idx[not_net] = idx

        state_size += n_fixup
        weights["has_fixup"] = torch.tensor([1], dtype=torch.int64)
    else:
        weights["has_fixup"] = torch.tensor([0], dtype=torch.int64)

    # Resolve output indices
    output_indices = []
    for bit in circuit.output_bits:
        if bit in net_to_idx:
            output_indices.append(net_to_idx[bit])
        else:
            raise ValueError(f"Output net {bit} not found in net mapping")

    weights["network_meta"] = torch.tensor(
        [num_inputs, num_outputs, num_real_layers, current_size],
        dtype=torch.int64,
    )
    weights["_output_indices"] = output_indices

    return weights


def _generate_sparse_linear_code() -> str:
    """Generate sparse_linear.py module."""
    return textwrap.dedent('''\
        """Sparse linear layer using COO format for verilog2nn compiled circuits."""

        import torch
        import torch.nn as nn


        class SparseLinear(nn.Module):
            """Linear layer stored in COO sparse format.

            Args:
                indices: (2, nnz) int64 tensor of (row, col) indices
                values: (nnz,) float64 tensor of non-zero values
                bias: (out_features,) float64 tensor
                shape: (out_features, in_features) tuple
            """

            def __init__(self, indices, values, bias, shape):
                super().__init__()
                out_features, in_features = int(shape[0]), int(shape[1])
                sparse_w = torch.sparse_coo_tensor(
                    indices, values, (out_features, in_features),
                ).coalesce()
                # Store as dense for reliable computation
                self.register_buffer("weight", sparse_w.to_dense())
                self.register_buffer("bias", bias)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)
    ''')


def _generate_model_code() -> str:
    """Generate model.py with SparseLinear + ReLU MLP architecture."""
    return textwrap.dedent('''\
        """Auto-generated PyTorch model for verilog2nn compiled circuit.

        Architecture: SparseLinear -> ReLU stacked MLP.
        Each layer compiles AND gates; NOT gates are folded into weights.
        """

        import torch
        import torch.nn as nn
        from safetensors.torch import load_file
        from sparse_linear import SparseLinear


        class VerilogNN(nn.Module):
            """Neural network equivalent of a Verilog combinational circuit.

            Architecture: Input -> [SparseLinear + ReLU] x N -> Output
            """

            def __init__(self, weights_path: str):
                super().__init__()
                data = load_file(weights_path)

                meta = data["network_meta"]
                self.num_inputs = meta[0].item()
                self.num_outputs = meta[1].item()
                num_layers = meta[2].item()
                self.init_state_size = meta[3].item()

                self.register_buffer(
                    "output_indices", data["output_indices"].long()
                )

                self.has_consts = data["has_consts"][0].item() == 1
                if self.has_consts:
                    self.register_buffer(
                        "const_values", data["const_values"].double()
                    )
                    self.register_buffer(
                        "const_indices", data["const_indices"].long()
                    )

                layers = []
                for i in range(num_layers):
                    layers.append(SparseLinear(
                        data[f"layer{i}.indices"],
                        data[f"layer{i}.values"],
                        data[f"layer{i}.bias"],
                        data[f"layer{i}.shape"],
                    ))
                self.layers = nn.ModuleList(layers)
                self.relu = nn.ReLU()

                self.has_fixup = data["has_fixup"][0].item() == 1
                if self.has_fixup:
                    self.fixup = SparseLinear(
                        data["fixup.indices"],
                        data["fixup.values"],
                        data["fixup.bias"],
                        data["fixup.shape"],
                    )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Forward pass. x shape: (batch, num_inputs), values in {0, 1}."""
                x = x.double()
                batch = x.shape[0]
                state = torch.zeros(
                    batch, self.init_state_size,
                    dtype=torch.float64, device=x.device,
                )
                state[:, :self.num_inputs] = x

                if self.has_consts:
                    state[:, self.const_indices] = self.const_values

                for layer in self.layers:
                    gate_out = self.relu(layer(state))
                    state = torch.cat([state, gate_out], dim=-1)

                if self.has_fixup:
                    fixup_out = self.fixup(state)
                    state = torch.cat([state, fixup_out], dim=-1)

                out = state[:, self.output_indices]
                return out.round().long()
    ''')


def _generate_inference_code(
    num_inputs: int,
    num_outputs: int,
    input_names: list[str],
    output_names: list[str],
) -> str:
    """Generate standalone inference script."""
    input_names_str = repr(input_names)
    output_names_str = repr(output_names)
    return textwrap.dedent(f'''\
        #!/usr/bin/env python3
        """Standalone inference script for verilog2nn compiled circuit.

        Usage:
            python inference.py <input_bits>
            python inference.py 0110  # For a 4-bit input circuit
        """

        import sys
        from pathlib import Path

        import torch

        sys.path.insert(0, str(Path(__file__).parent))
        from model import VerilogNN

        NUM_INPUTS = {num_inputs}
        NUM_OUTPUTS = {num_outputs}
        INPUT_NAMES = {input_names_str}
        OUTPUT_NAMES = {output_names_str}


        def load_model() -> VerilogNN:
            weights_path = str(Path(__file__).parent / "weights.safetensors")
            model = VerilogNN(weights_path)
            model.eval()
            return model


        def infer(model: VerilogNN, input_bits: list[int]) -> list[int]:
            """Run inference on a single input vector."""
            assert len(input_bits) == NUM_INPUTS, (
                f"Expected {{NUM_INPUTS}} input bits, got {{len(input_bits)}}"
            )
            x = torch.tensor([input_bits], dtype=torch.float64)
            with torch.no_grad():
                y = model(x)
            return y[0].tolist()


        def main():
            if len(sys.argv) < 2:
                print(f"Usage: {{sys.argv[0]}} <input_bits>")
                print(f"  Input: {{NUM_INPUTS}} bits ({{', '.join(INPUT_NAMES)}})")
                print(f"  Output: {{NUM_OUTPUTS}} bits ({{', '.join(OUTPUT_NAMES)}})")
                sys.exit(1)

            bits_str = sys.argv[1]
            if len(bits_str) != NUM_INPUTS:
                print(f"Error: expected {{NUM_INPUTS}} bits, got {{len(bits_str)}}")
                sys.exit(1)

            input_bits = [int(b) for b in bits_str]
            model = load_model()
            output = infer(model, input_bits)

            print("Inputs:")
            for name, val in zip(INPUT_NAMES, input_bits):
                print(f"  {{name}} = {{val}}")
            print("Outputs:")
            for name, val in zip(OUTPUT_NAMES, output):
                print(f"  {{name}} = {{int(val)}}")


        if __name__ == "__main__":
            main()
    ''')
