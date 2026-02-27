"""Map gate-level circuits to PyTorch neural networks.

Gate mapping (inputs are {0,1}):
    NOT(x)   = 1 - x                     (linear)
    AND(a,b) = ReLU(a+b-1) - ReLU(a+b-2) (ReLU + linear, but since a+b <= 2, ReLU(a+b-2)=0 always, simplifies to ReLU(a+b-1))
    OR(a,b)  = (a+b) - ReLU(a+b-1)       (ReLU + linear)
    XOR(a,b) = (a+b) - 2*ReLU(a+b-1)     (ReLU + linear)

Each topological layer is compiled to a pair of linear transforms:
1. Pre-ReLU linear: computes weighted sums for gates needing ReLU
2. Post-ReLU linear: combines ReLU outputs with direct outputs

Wire routing between layers is handled implicitly through the weight matrices.

For large circuits (>1000 gates), sparse COO format is used to avoid
creating huge dense matrices. The model uses indexed gather/scatter
operations for efficient inference.
"""

import textwrap
from pathlib import Path

import torch
from safetensors.torch import save_file

from verilog2nn.netlist_parser import Circuit, Gate

# Threshold for switching to sparse representation
SPARSE_THRESHOLD = 1000


def compile_to_nn(
    circuit: Circuit,
    layers: list[list[Gate]],
    output_dir: Path,
) -> None:
    """Compile circuit to PyTorch NN, saving model code + weights + inference script.

    The network architecture:
    - Input: binary vector of size num_inputs
    - For each gate layer: Linear -> ReLU -> Linear (with passthrough for wires)
    - Output: binary vector of size num_outputs

    Net tracking: We maintain a mapping from net_id -> (vector_index, source)
    where source tracks whether the net value lives in the current "state" vector.
    """
    num_inputs = len(circuit.input_bits)
    num_outputs = len(circuit.output_bits)
    total_gates = len(circuit.gates)
    use_sparse = total_gates > SPARSE_THRESHOLD

    # Assign indices: net_id -> index in current state vector
    # Start with primary inputs
    net_to_idx: dict[int, int] = {}
    for i, bit in enumerate(circuit.input_bits):
        net_to_idx[bit] = i

    # Also assign indices for constant nets
    current_size = num_inputs
    for net_id, val in circuit.const_nets.items():
        net_to_idx[net_id] = current_size
        current_size += 1

    # Constant values vector (appended to input)
    const_values = []
    const_indices = []
    for net_id, val in circuit.const_nets.items():
        const_values.append(float(val))
        const_indices.append(net_to_idx[net_id])

    if use_sparse:
        weights = _compile_sparse(
            circuit, layers, net_to_idx, current_size,
            num_inputs, num_outputs,
        )
    else:
        weights = _compile_dense(
            circuit, layers, net_to_idx, current_size,
            num_inputs, num_outputs,
        )

    # Common metadata
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

    # Sparse flag
    weights["is_sparse"] = torch.tensor(
        [1 if use_sparse else 0], dtype=torch.int64,
    )

    save_file(weights, output_dir / "weights.safetensors")

    # Generate model.py
    model_code = _generate_model_code(
        num_inputs, num_outputs,
        weights["network_meta"][2].item(),
        len(circuit.const_nets) > 0,
    )
    (output_dir / "model.py").write_text(model_code)

    # Generate inference.py
    inference_code = _generate_inference_code(
        num_inputs, num_outputs,
        circuit.input_names, circuit.output_names,
    )
    (output_dir / "inference.py").write_text(inference_code)


def _compile_dense(
    circuit: Circuit,
    layers: list[list[Gate]],
    net_to_idx: dict[int, int],
    current_size: int,
    num_inputs: int,
    num_outputs: int,
) -> dict:
    """Original dense matrix compilation for small circuits."""
    layer_params = []

    for layer_gates in layers:
        n_gates = len(layer_gates)
        state_size = current_size

        relu_gates = [g for g in layer_gates if g.gate_type != "NOT"]
        not_gates = [g for g in layer_gates if g.gate_type == "NOT"]
        n_relu = len(relu_gates)

        W_pre = torch.zeros(n_relu, state_size, dtype=torch.float64)
        b_pre = torch.zeros(n_relu, dtype=torch.float64)

        for i, gate in enumerate(relu_gates):
            for inp_net in gate.input_nets:
                idx = net_to_idx[inp_net]
                W_pre[i, idx] = 1.0
            b_pre[i] = -1.0

        post_input_size = state_size + n_relu
        W_post = torch.zeros(n_gates, post_input_size, dtype=torch.float64)
        b_post = torch.zeros(n_gates, dtype=torch.float64)

        gate_output_start = current_size

        for gi, gate in enumerate(not_gates):
            inp_net = gate.input_nets[0]
            inp_idx = net_to_idx[inp_net]
            W_post[gi, inp_idx] = -1.0
            b_post[gi] = 1.0

        not_count = len(not_gates)
        for ri, gate in enumerate(relu_gates):
            gi = not_count + ri
            relu_idx = state_size + ri

            if gate.gate_type == "AND":
                W_post[gi, relu_idx] = 1.0
            elif gate.gate_type == "OR":
                for inp_net in gate.input_nets:
                    inp_idx = net_to_idx[inp_net]
                    W_post[gi, inp_idx] = 1.0
                W_post[gi, relu_idx] = -1.0
            elif gate.gate_type == "XOR":
                for inp_net in gate.input_nets:
                    inp_idx = net_to_idx[inp_net]
                    W_post[gi, inp_idx] = 1.0
                W_post[gi, relu_idx] = -2.0

        all_gates_ordered = not_gates + relu_gates
        for gi, gate in enumerate(all_gates_ordered):
            net_to_idx[gate.output_net] = current_size
            current_size += 1

        layer_params.append({
            "W_pre": W_pre,
            "b_pre": b_pre,
            "W_post": W_post,
            "b_post": b_post,
            "state_size": state_size,
            "n_relu": n_relu,
            "n_gates": n_gates,
            "gate_output_start": gate_output_start,
        })

    output_indices = []
    for bit in circuit.output_bits:
        if bit in net_to_idx:
            output_indices.append(net_to_idx[bit])
        else:
            raise ValueError(f"Output net {bit} not found in net mapping")

    weights = {}
    for i, lp in enumerate(layer_params):
        weights[f"layer_{i}_W_pre"] = lp["W_pre"]
        weights[f"layer_{i}_b_pre"] = lp["b_pre"]
        weights[f"layer_{i}_W_post"] = lp["W_post"]
        weights[f"layer_{i}_b_post"] = lp["b_post"]

    meta_list = []
    for lp in layer_params:
        meta_list.extend([
            lp["state_size"],
            lp["n_relu"],
            lp["n_gates"],
            lp["gate_output_start"],
        ])
    weights["layer_meta"] = torch.tensor(meta_list, dtype=torch.int64)
    weights["network_meta"] = torch.tensor(
        [num_inputs, num_outputs, len(layer_params), current_size],
        dtype=torch.int64,
    )
    weights["_output_indices"] = output_indices
    return weights


def _compile_sparse(
    circuit: Circuit,
    layers: list[list[Gate]],
    net_to_idx: dict[int, int],
    current_size: int,
    num_inputs: int,
    num_outputs: int,
) -> dict:
    """Sparse compilation for large circuits.

    Instead of dense matrices, stores per-gate operation descriptors:
    - gate_type: 0=NOT, 1=AND, 2=OR, 3=XOR
    - input indices (up to 2)
    - output index in state vector

    The model evaluates gates directly using gather/scatter.
    """
    # Gate type encoding
    TYPE_MAP = {"NOT": 0, "AND": 1, "OR": 2, "XOR": 3}

    # Flatten all gates in topological order, recording per-gate info
    gate_types = []   # int per gate
    gate_inp0 = []    # first input index
    gate_inp1 = []    # second input index (0 for NOT, unused)
    gate_out = []     # output index in state vector

    # Also track layer boundaries for batched evaluation
    layer_sizes = []

    for layer_gates in layers:
        layer_sizes.append(len(layer_gates))
        for gate in layer_gates:
            gate_types.append(TYPE_MAP[gate.gate_type])
            gate_inp0.append(net_to_idx[gate.input_nets[0]])
            if len(gate.input_nets) > 1:
                gate_inp1.append(net_to_idx[gate.input_nets[1]])
            else:
                gate_inp1.append(0)  # unused for NOT
            # Assign output index
            net_to_idx[gate.output_net] = current_size
            gate_out.append(current_size)
            current_size += 1

    output_indices = []
    for bit in circuit.output_bits:
        if bit in net_to_idx:
            output_indices.append(net_to_idx[bit])
        else:
            raise ValueError(f"Output net {bit} not found in net mapping")

    weights = {
        "gate_types": torch.tensor(gate_types, dtype=torch.int64),
        "gate_inp0": torch.tensor(gate_inp0, dtype=torch.int64),
        "gate_inp1": torch.tensor(gate_inp1, dtype=torch.int64),
        "gate_out": torch.tensor(gate_out, dtype=torch.int64),
        "layer_sizes": torch.tensor(layer_sizes, dtype=torch.int64),
        "network_meta": torch.tensor(
            [num_inputs, num_outputs, len(layers), current_size],
            dtype=torch.int64,
        ),
        # Unused but kept for API compatibility
        "layer_meta": torch.tensor([0], dtype=torch.int64),
        "_output_indices": output_indices,
    }
    return weights


def _generate_model_code(
    num_inputs: int,
    num_outputs: int,
    num_layers: int,
    has_consts: bool,
) -> str:
    """Generate PyTorch nn.Module code for the compiled network."""
    return textwrap.dedent(f'''\
        """Auto-generated PyTorch model for verilog2nn compiled circuit."""

        import torch
        import torch.nn as nn
        from safetensors.torch import load_file


        class VerilogNN(nn.Module):
            """Neural network equivalent of a Verilog combinational circuit."""

            def __init__(self, weights_path: str):
                super().__init__()
                data = load_file(weights_path)

                meta = data["network_meta"]
                self.num_inputs = meta[0].item()
                self.num_outputs = meta[1].item()
                self.num_layers = meta[2].item()
                self.total_state_size = meta[3].item()

                self.output_indices = data["output_indices"].long()
                self.has_consts = data["has_consts"][0].item() == 1
                self.is_sparse = data.get("is_sparse", torch.tensor([0]))[0].item() == 1

                if self.has_consts:
                    self.const_values = data["const_values"].double()
                    self.const_indices = data["const_indices"].long()

                if self.is_sparse:
                    self._init_sparse(data)
                else:
                    self._init_dense(data)

            def _init_sparse(self, data):
                self.gate_types = data["gate_types"].long()
                self.gate_inp0 = data["gate_inp0"].long()
                self.gate_inp1 = data["gate_inp1"].long()
                self.gate_out = data["gate_out"].long()
                self.layer_sizes_list = data["layer_sizes"].long().tolist()

                # Precompute per-type masks for vectorized evaluation
                self.not_mask = self.gate_types == 0
                self.and_mask = self.gate_types == 1
                self.or_mask = self.gate_types == 2
                self.xor_mask = self.gate_types == 3

                # Precompute layer boundaries (cumsum)
                self._layer_offsets = []
                offset = 0
                for s in self.layer_sizes_list:
                    self._layer_offsets.append((offset, offset + s))
                    offset += s

            def _init_dense(self, data):
                layer_meta = data["layer_meta"]
                self.layers_info = []
                self.W_pres = nn.ParameterList()
                self.b_pres = nn.ParameterList()
                self.W_posts = nn.ParameterList()
                self.b_posts = nn.ParameterList()

                for i in range(self.num_layers):
                    base = i * 4
                    info = {{
                        "state_size": layer_meta[base].item(),
                        "n_relu": layer_meta[base + 1].item(),
                        "n_gates": layer_meta[base + 2].item(),
                        "gate_output_start": layer_meta[base + 3].item(),
                    }}
                    self.layers_info.append(info)

                    W_pre = data[f"layer_{{i}}_W_pre"].double()
                    b_pre = data[f"layer_{{i}}_b_pre"].double()
                    W_post = data[f"layer_{{i}}_W_post"].double()
                    b_post = data[f"layer_{{i}}_b_post"].double()

                    self.W_pres.append(nn.Parameter(W_pre, requires_grad=False))
                    self.b_pres.append(nn.Parameter(b_pre, requires_grad=False))
                    self.W_posts.append(nn.Parameter(W_post, requires_grad=False))
                    self.b_posts.append(nn.Parameter(b_post, requires_grad=False))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Forward pass. x shape: (batch, num_inputs), values in {{0, 1}}.

                Returns: (batch, num_outputs), values in {{0, 1}}.
                """
                if self.is_sparse:
                    return self._forward_sparse(x)
                else:
                    return self._forward_dense(x)

            def _forward_sparse(self, x: torch.Tensor) -> torch.Tensor:
                x = x.double()
                batch = x.shape[0]
                state = torch.zeros(batch, self.total_state_size, dtype=torch.float64, device=x.device)
                state[:, :self.num_inputs] = x

                if self.has_consts:
                    for ci in range(len(self.const_indices)):
                        state[:, self.const_indices[ci]] = self.const_values[ci]

                # Process layer by layer
                for start, end in self._layer_offsets:
                    types = self.gate_types[start:end]
                    inp0 = self.gate_inp0[start:end]
                    inp1 = self.gate_inp1[start:end]
                    out_idx = self.gate_out[start:end]

                    a = state[:, inp0]  # (batch, n_gates_in_layer)
                    b = state[:, inp1]  # (batch, n_gates_in_layer)

                    # Compute gate outputs vectorized by type
                    result = torch.zeros(batch, end - start, dtype=torch.float64, device=x.device)

                    not_m = types == 0
                    and_m = types == 1
                    or_m = types == 2
                    xor_m = types == 3

                    if not_m.any():
                        result[:, not_m] = 1.0 - a[:, not_m]
                    if and_m.any():
                        result[:, and_m] = torch.relu(a[:, and_m] + b[:, and_m] - 1.0)
                    if or_m.any():
                        ab = a[:, or_m] + b[:, or_m]
                        result[:, or_m] = ab - torch.relu(ab - 1.0)
                    if xor_m.any():
                        ab = a[:, xor_m] + b[:, xor_m]
                        result[:, xor_m] = ab - 2.0 * torch.relu(ab - 1.0)

                    state[:, out_idx] = result

                out = state[:, self.output_indices]
                return out.round().long()

            def _forward_dense(self, x: torch.Tensor) -> torch.Tensor:
                x = x.double()
                batch = x.shape[0]

                state = torch.zeros(batch, self.total_state_size, dtype=torch.float64, device=x.device)
                state[:, :self.num_inputs] = x

                if self.has_consts:
                    for ci in range(len(self.const_indices)):
                        state[:, self.const_indices[ci]] = self.const_values[ci]

                for i, info in enumerate(self.layers_info):
                    ss = info["state_size"]
                    n_relu = info["n_relu"]
                    n_gates = info["n_gates"]
                    gs = info["gate_output_start"]

                    current_state = state[:, :ss]

                    if n_relu > 0:
                        h = current_state @ self.W_pres[i].T + self.b_pres[i]
                        r = torch.relu(h)
                        combined = torch.cat([current_state, r], dim=1)
                    else:
                        combined = current_state

                    gate_out = combined @ self.W_posts[i].T + self.b_posts[i]
                    state[:, gs:gs + n_gates] = gate_out

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

        Input bits are MSB-first by default, matching Verilog [N-1:0] convention.
        """

        import sys
        from pathlib import Path

        import torch

        # Add model directory to path
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
