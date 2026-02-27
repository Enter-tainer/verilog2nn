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

    # Build layer weights
    # For each layer, we produce matrices for a 2-step computation:
    #   h = W_pre @ state + b_pre   (pre-activation)
    #   r = ReLU(h)                  (activation)
    #   new_nets = W_post @ [state; r] + b_post  (post-activation, produces gate outputs)
    # Then state gets updated: pass-through old nets + new gate outputs

    layer_params = []

    for layer_gates in layers:
        n_gates = len(layer_gates)
        state_size = current_size

        # Each gate that needs ReLU produces one pre-activation value
        # NOT gates don't need ReLU
        relu_gates = [g for g in layer_gates if g.gate_type != "NOT"]
        not_gates = [g for g in layer_gates if g.gate_type == "NOT"]
        n_relu = len(relu_gates)

        # Pre-activation: compute a+b-1 for each ReLU gate
        # Shape: (n_relu, state_size), bias: (n_relu,)
        W_pre = torch.zeros(n_relu, state_size, dtype=torch.float64)
        b_pre = torch.zeros(n_relu, dtype=torch.float64)

        for i, gate in enumerate(relu_gates):
            for inp_net in gate.input_nets:
                idx = net_to_idx[inp_net]
                W_pre[i, idx] = 1.0
            b_pre[i] = -1.0  # a+b-1

        # Post-activation: combine state and ReLU outputs to produce gate outputs
        # Input to post: [state (state_size), relu_outputs (n_relu)]
        # Output: gate output values for each gate in this layer
        post_input_size = state_size + n_relu
        W_post = torch.zeros(n_gates, post_input_size, dtype=torch.float64)
        b_post = torch.zeros(n_gates, dtype=torch.float64)

        gate_output_start = current_size

        # Process NOT gates (linear only, no ReLU needed)
        for gi, gate in enumerate(not_gates):
            out_idx = gi  # position within this layer's output
            inp_net = gate.input_nets[0]
            inp_idx = net_to_idx[inp_net]
            # NOT(x) = 1 - x
            W_post[gi, inp_idx] = -1.0  # -x from state
            b_post[gi] = 1.0  # +1

        # Process ReLU gates
        not_count = len(not_gates)
        for ri, gate in enumerate(relu_gates):
            gi = not_count + ri  # position within this layer's output
            relu_idx = state_size + ri  # position in post input (after state)

            if gate.gate_type == "AND":
                # AND(a,b) = ReLU(a+b-1)
                # (since a+b <= 2, ReLU(a+b-2) = 0 always when inputs are binary)
                W_post[gi, relu_idx] = 1.0  # +ReLU(a+b-1)

            elif gate.gate_type == "OR":
                # OR(a,b) = (a+b) - ReLU(a+b-1)
                for inp_net in gate.input_nets:
                    inp_idx = net_to_idx[inp_net]
                    W_post[gi, inp_idx] = 1.0  # +a, +b from state
                W_post[gi, relu_idx] = -1.0  # -ReLU(a+b-1)

            elif gate.gate_type == "XOR":
                # XOR(a,b) = (a+b) - 2*ReLU(a+b-1)
                for inp_net in gate.input_nets:
                    inp_idx = net_to_idx[inp_net]
                    W_post[gi, inp_idx] = 1.0  # +a, +b from state
                W_post[gi, relu_idx] = -2.0  # -2*ReLU(a+b-1)

        # Assign net indices for gate outputs
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

    # Build output extraction indices
    output_indices = []
    for bit in circuit.output_bits:
        if bit in net_to_idx:
            output_indices.append(net_to_idx[bit])
        else:
            raise ValueError(f"Output net {bit} not found in net mapping")

    # Save weights
    weights = {}
    for i, lp in enumerate(layer_params):
        weights[f"layer_{i}_W_pre"] = lp["W_pre"]
        weights[f"layer_{i}_b_pre"] = lp["b_pre"]
        weights[f"layer_{i}_W_post"] = lp["W_post"]
        weights[f"layer_{i}_b_post"] = lp["b_post"]

    # Save metadata as 1D tensors
    weights["output_indices"] = torch.tensor(output_indices, dtype=torch.int64)
    weights["const_values"] = torch.tensor(
        const_values if const_values else [0.0], dtype=torch.float64
    )
    weights["const_indices"] = torch.tensor(
        const_indices if const_indices else [0], dtype=torch.int64
    )

    # Metadata for reconstruction
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
    if const_values:
        weights["has_consts"] = torch.tensor([1], dtype=torch.int64)
    else:
        weights["has_consts"] = torch.tensor([0], dtype=torch.int64)

    save_file(weights, output_dir / "weights.safetensors")

    # Generate model.py
    model_code = _generate_model_code(
        num_inputs, num_outputs, len(layer_params), len(circuit.const_nets) > 0
    )
    (output_dir / "model.py").write_text(model_code)

    # Generate inference.py
    inference_code = _generate_inference_code(
        num_inputs,
        num_outputs,
        circuit.input_names,
        circuit.output_names,
    )
    (output_dir / "inference.py").write_text(inference_code)


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

                if self.has_consts:
                    self.const_values = data["const_values"].double()
                    self.const_indices = data["const_indices"].long()

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
                x = x.double()
                batch = x.shape[0]

                # Initialize state with inputs
                state = torch.zeros(batch, self.total_state_size, dtype=torch.float64, device=x.device)
                state[:, :self.num_inputs] = x

                # Set constant values
                if self.has_consts:
                    for ci in range(len(self.const_indices)):
                        state[:, self.const_indices[ci]] = self.const_values[ci]

                # Process each layer
                for i, info in enumerate(self.layers_info):
                    ss = info["state_size"]
                    n_relu = info["n_relu"]
                    n_gates = info["n_gates"]
                    gs = info["gate_output_start"]

                    current_state = state[:, :ss]

                    # Pre-activation + ReLU
                    if n_relu > 0:
                        h = current_state @ self.W_pres[i].T + self.b_pres[i]
                        r = torch.relu(h)
                        # Post-activation
                        combined = torch.cat([current_state, r], dim=1)
                    else:
                        combined = current_state

                    gate_out = combined @ self.W_posts[i].T + self.b_posts[i]
                    state[:, gs:gs + n_gates] = gate_out

                # Extract outputs
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
