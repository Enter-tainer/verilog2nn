"""Map gate-level circuits to PyTorch neural networks (pure MLP architecture).

Gate mapping (inputs are {0,1}):
    NOT(x)   = 1 - x                     (linear, no ReLU needed)
    AND(a,b) = ReLU(a+b-1)               (pre: a+b-1, post: take relu output)
    OR(a,b)  = (a+b) - ReLU(a+b-1)       (pre: a+b-1, post: (a+b) - relu)
    XOR(a,b) = (a+b) - 2*ReLU(a+b-1)     (pre: a+b-1, post: (a+b) - 2*relu)

Each topological layer becomes a GateBlock with two nn.Linear layers:
    Linear_pre:  state_size -> n_relu   (computes a+b-1 for AND/OR/XOR gates)
    ReLU
    Linear_post: (state_size + n_relu) -> n_gates  (combines state & relu outputs)

NOT gates are purely linear and handled in the post layer only.
The model stacks GateBlocks for a clean standard MLP-like interface.
Weights are stored as dense matrices via safetensors.
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
    """Compile circuit to PyTorch NN, saving model code + weights + inference script."""
    num_inputs = len(circuit.input_bits)
    num_outputs = len(circuit.output_bits)

    # Assign indices: net_id -> index in current state vector
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
    """Compile all layers to dense W_pre/b_pre/W_post/b_post per layer.

    For each topological layer:
    - Separate gates into relu_gates (AND/OR/XOR) and not_gates (NOT)
    - Build W_pre (state_size x n_relu): gathers a+b for relu gates, bias=-1
    - Build W_post ((state_size + n_relu) x n_gates): combines state+relu -> output
    """
    weights: dict = {}
    state_size = current_size

    for layer_idx, layer_gates in enumerate(layers):
        relu_gates = [g for g in layer_gates if g.gate_type != "NOT"]
        not_gates = [g for g in layer_gates if g.gate_type == "NOT"]
        n_relu = len(relu_gates)
        n_gates = len(layer_gates)

        # --- W_pre: state_size -> n_relu ---
        # For each relu gate: output = a + b (then bias = -1, then ReLU)
        W_pre = torch.zeros(n_relu, state_size, dtype=torch.float64)
        b_pre = torch.full((n_relu,), -1.0, dtype=torch.float64)

        for j, gate in enumerate(relu_gates):
            idx_a = net_to_idx[gate.input_nets[0]]
            idx_b = net_to_idx[gate.input_nets[1]]
            W_pre[j, idx_a] = 1.0
            W_pre[j, idx_b] = 1.0

        # --- W_post: (state_size + n_relu) -> n_gates ---
        # Input to post layer is [state; relu_output]
        post_in_size = state_size + n_relu
        W_post = torch.zeros(n_gates, post_in_size, dtype=torch.float64)
        b_post = torch.zeros(n_gates, dtype=torch.float64)

        # We need a consistent ordering: all gates in layer_gates order
        # Build a mapping from gate -> output index in this layer
        # relu_idx tracks which relu output corresponds to which relu gate
        relu_idx_map = {}
        for j, gate in enumerate(relu_gates):
            relu_idx_map[id(gate)] = j

        for out_j, gate in enumerate(layer_gates):
            if gate.gate_type == "NOT":
                # NOT(x) = 1 - x
                idx_x = net_to_idx[gate.input_nets[0]]
                W_post[out_j, idx_x] = -1.0  # -x from state
                b_post[out_j] = 1.0  # + 1
            elif gate.gate_type == "AND":
                # AND(a,b) = ReLU(a+b-1)
                # post: just take the relu output directly
                relu_j = relu_idx_map[id(gate)]
                W_post[out_j, state_size + relu_j] = 1.0
            elif gate.gate_type == "OR":
                # OR(a,b) = (a+b) - ReLU(a+b-1)
                # post: state[a] + state[b] - relu[j]
                idx_a = net_to_idx[gate.input_nets[0]]
                idx_b = net_to_idx[gate.input_nets[1]]
                W_post[out_j, idx_a] = 1.0
                W_post[out_j, idx_b] = 1.0
                relu_j = relu_idx_map[id(gate)]
                W_post[out_j, state_size + relu_j] = -1.0
            elif gate.gate_type == "XOR":
                # XOR(a,b) = (a+b) - 2*ReLU(a+b-1)
                # post: state[a] + state[b] - 2*relu[j]
                idx_a = net_to_idx[gate.input_nets[0]]
                idx_b = net_to_idx[gate.input_nets[1]]
                W_post[out_j, idx_a] = 1.0
                W_post[out_j, idx_b] = 1.0
                relu_j = relu_idx_map[id(gate)]
                W_post[out_j, state_size + relu_j] = -2.0

        weights[f"layer{layer_idx}.W_pre"] = W_pre
        weights[f"layer{layer_idx}.b_pre"] = b_pre
        weights[f"layer{layer_idx}.W_post"] = W_post
        weights[f"layer{layer_idx}.b_post"] = b_post

        # Assign output indices for this layer's gates
        # In forward: new_state = cat(state, gate_out)
        # So gate_out[j] is at index state_size + j in the new state
        for out_j, gate in enumerate(layer_gates):
            net_to_idx[gate.output_net] = state_size + out_j

        # Next layer's state_size = current + n_gates (relu is transient)
        state_size = state_size + n_gates

    output_indices = []
    for bit in circuit.output_bits:
        if bit in net_to_idx:
            output_indices.append(net_to_idx[bit])
        else:
            raise ValueError(f"Output net {bit} not found in net mapping")

    weights["network_meta"] = torch.tensor(
        [num_inputs, num_outputs, len(layers), current_size],
        dtype=torch.int64,
    )
    weights["_output_indices"] = output_indices

    return weights


def _generate_model_code() -> str:
    """Generate PyTorch nn.Module code for the compiled network (pure MLP)."""
    return textwrap.dedent('''\
        """Auto-generated PyTorch model for verilog2nn compiled circuit.

        Architecture: pure MLP (nn.Linear + nn.ReLU) stacked in GateBlocks.
        """

        import torch
        import torch.nn as nn
        from safetensors.torch import load_file


        class GateBlock(nn.Module):
            """A single topological layer of logic gates as Linear+ReLU+Linear.

            Forward:
                relu_in = Linear_pre(state)      # state_size -> n_relu
                relu_out = ReLU(relu_in)
                combined = cat(state, relu_out)   # state_size + n_relu
                gate_out = Linear_post(combined)  # -> n_gates
                new_state = cat(state, gate_out)   # state_size + n_gates
            """

            def __init__(self, W_pre, b_pre, W_post, b_post, state_size, n_relu):
                super().__init__()
                self.state_size = state_size
                self.n_relu = n_relu

                self.linear_pre = nn.Linear(state_size, n_relu, dtype=torch.float64)
                self.relu = nn.ReLU()
                n_gates = W_post.shape[0]
                self.linear_post = nn.Linear(
                    state_size + n_relu, n_gates, dtype=torch.float64,
                )

                with torch.no_grad():
                    self.linear_pre.weight.copy_(W_pre)
                    self.linear_pre.bias.copy_(b_pre)
                    self.linear_post.weight.copy_(W_post)
                    self.linear_post.bias.copy_(b_post)

            def forward(self, state):
                relu_out = self.relu(self.linear_pre(state))
                combined = torch.cat([state, relu_out], dim=-1)
                gate_out = self.linear_post(combined)
                return torch.cat([state, gate_out], dim=-1)


        class VerilogNN(nn.Module):
            """Neural network equivalent of a Verilog combinational circuit.

            Architecture: Input -> [GateBlock x N] -> Output
            Each GateBlock is Linear -> ReLU -> Linear (standard MLP pattern).
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

                blocks = []
                state_size = self.init_state_size
                for i in range(num_layers):
                    W_pre = data[f"layer{i}.W_pre"]
                    b_pre = data[f"layer{i}.b_pre"]
                    W_post = data[f"layer{i}.W_post"]
                    b_post = data[f"layer{i}.b_post"]
                    n_relu = W_pre.shape[0]
                    blocks.append(
                        GateBlock(W_pre, b_pre, W_post, b_post, state_size, n_relu)
                    )
                    n_gates = W_post.shape[0]
                    state_size += n_gates
                self.blocks = nn.ModuleList(blocks)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Forward pass. x shape: (batch, num_inputs), values in {0, 1}.

                Returns: (batch, num_outputs), values in {0, 1}.
                """
                x = x.double()
                batch = x.shape[0]
                state = torch.zeros(
                    batch, self.init_state_size,
                    dtype=torch.float64, device=x.device,
                )
                state[:, :self.num_inputs] = x

                if self.has_consts:
                    state[:, self.const_indices] = self.const_values

                for block in self.blocks:
                    state = block(state)

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
