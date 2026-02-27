"""Map gate-level circuits to PyTorch neural networks.

Gate mapping (inputs are {0,1}):
    NOT(x)   = 1 - x                     (linear)
    AND(a,b) = ReLU(a+b-1)               (since a+b <= 2, ReLU(a+b-2)=0 always)
    OR(a,b)  = (a+b) - ReLU(a+b-1)       (ReLU + linear)
    XOR(a,b) = (a+b) - 2*ReLU(a+b-1)     (ReLU + linear)

Each topological layer becomes a SparseGateLayer that:
1. Gathers input values from the state vector
2. Computes gate outputs (NOT/AND/OR/XOR) vectorized by type
3. Scatters results back to the state vector

The model stacks layers with nn.Sequential for a clean PyTorch interface.
Weights are stored in sparse COO format via safetensors.
"""

import textwrap
from pathlib import Path

import torch
from safetensors.torch import save_file

from verilog2nn.netlist_parser import Circuit, Gate

# Gate type encoding
TYPE_MAP = {"NOT": 0, "AND": 1, "OR": 2, "XOR": 3}


def compile_to_nn(
    circuit: Circuit,
    layers: list[list[Gate]],
    output_dir: Path,
) -> None:
    """Compile circuit to PyTorch NN, saving model code + weights + inference script.

    The network architecture:
    - Input: binary vector of size num_inputs
    - For each gate layer: SparseGateLayer (gather -> gate ops -> scatter)
    - Output: binary vector of size num_outputs

    Net tracking: We maintain a mapping from net_id -> index in state vector.
    """
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
    model_code = _generate_model_code(
        num_inputs, num_outputs,
        weights["network_meta"][2].item(),
    )
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
    """Compile all layers to sparse COO format.

    Per-gate descriptors:
    - gate_type: 0=NOT, 1=AND, 2=OR, 3=XOR
    - input indices (up to 2)
    - output index in state vector
    """
    gate_types = []
    gate_inp0 = []
    gate_inp1 = []
    gate_out = []
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
            net_to_idx[gate.output_net] = current_size
            gate_out.append(current_size)
            current_size += 1

    output_indices = []
    for bit in circuit.output_bits:
        if bit in net_to_idx:
            output_indices.append(net_to_idx[bit])
        else:
            raise ValueError(f"Output net {bit} not found in net mapping")

    return {
        "gate_types": torch.tensor(gate_types, dtype=torch.int64),
        "gate_inp0": torch.tensor(gate_inp0, dtype=torch.int64),
        "gate_inp1": torch.tensor(gate_inp1, dtype=torch.int64),
        "gate_out": torch.tensor(gate_out, dtype=torch.int64),
        "layer_sizes": torch.tensor(layer_sizes, dtype=torch.int64),
        "network_meta": torch.tensor(
            [num_inputs, num_outputs, len(layers), current_size],
            dtype=torch.int64,
        ),
        "_output_indices": output_indices,
    }


def _generate_model_code(
    num_inputs: int,
    num_outputs: int,
    num_layers: int,
) -> str:
    """Generate PyTorch nn.Module code for the compiled network."""
    return textwrap.dedent('''\
        """Auto-generated PyTorch model for verilog2nn compiled circuit."""

        import torch
        import torch.nn as nn
        from safetensors.torch import load_file


        class SparseGateLayer(nn.Module):
            """A single layer of logic gates evaluated via sparse gather/scatter.

            Each layer gathers input values from the state vector, computes
            gate outputs (NOT/AND/OR/XOR) vectorized by type, and scatters
            results back to the state vector.
            """

            def __init__(
                self,
                gate_types: torch.Tensor,
                gate_inp0: torch.Tensor,
                gate_inp1: torch.Tensor,
                gate_out: torch.Tensor,
            ):
                super().__init__()
                self.register_buffer("gate_types", gate_types)
                self.register_buffer("gate_inp0", gate_inp0)
                self.register_buffer("gate_inp1", gate_inp1)
                self.register_buffer("gate_out", gate_out)

            def forward(self, state: torch.Tensor) -> torch.Tensor:
                """Apply gate operations and update state in-place.

                Args:
                    state: (batch, total_state_size) tensor

                Returns:
                    Updated state tensor.
                """
                a = state[:, self.gate_inp0]
                b = state[:, self.gate_inp1]

                n = self.gate_types.shape[0]
                result = torch.zeros(
                    state.shape[0], n,
                    dtype=state.dtype, device=state.device,
                )

                not_m = self.gate_types == 0
                and_m = self.gate_types == 1
                or_m = self.gate_types == 2
                xor_m = self.gate_types == 3

                if not_m.any():
                    result[:, not_m] = 1.0 - a[:, not_m]
                if and_m.any():
                    result[:, and_m] = torch.relu(
                        a[:, and_m] + b[:, and_m] - 1.0
                    )
                if or_m.any():
                    ab = a[:, or_m] + b[:, or_m]
                    result[:, or_m] = ab - torch.relu(ab - 1.0)
                if xor_m.any():
                    ab = a[:, xor_m] + b[:, xor_m]
                    result[:, xor_m] = ab - 2.0 * torch.relu(ab - 1.0)

                state = state.clone()
                state[:, self.gate_out] = result
                return state


        class VerilogNN(nn.Module):
            """Neural network equivalent of a Verilog combinational circuit.

            Architecture:
                Input -> [SparseGateLayer x N] -> Output
                Stacked via nn.Sequential.
            """

            def __init__(self, weights_path: str):
                super().__init__()
                data = load_file(weights_path)

                meta = data["network_meta"]
                self.num_inputs = meta[0].item()
                self.num_outputs = meta[1].item()
                self.num_layers = meta[2].item()
                self.total_state_size = meta[3].item()

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

                # Build layer stack from sparse gate descriptors
                gate_types = data["gate_types"].long()
                gate_inp0 = data["gate_inp0"].long()
                gate_inp1 = data["gate_inp1"].long()
                gate_out = data["gate_out"].long()
                layer_sizes = data["layer_sizes"].long().tolist()

                gate_layers = []
                offset = 0
                for size in layer_sizes:
                    gate_layers.append(SparseGateLayer(
                        gate_types=gate_types[offset:offset + size],
                        gate_inp0=gate_inp0[offset:offset + size],
                        gate_inp1=gate_inp1[offset:offset + size],
                        gate_out=gate_out[offset:offset + size],
                    ))
                    offset += size

                self.gate_layers = nn.Sequential(*gate_layers)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Forward pass. x shape: (batch, num_inputs), values in {0, 1}.

                Returns: (batch, num_outputs), values in {0, 1}.
                """
                x = x.double()
                batch = x.shape[0]
                state = torch.zeros(
                    batch, self.total_state_size,
                    dtype=torch.float64, device=x.device,
                )
                state[:, :self.num_inputs] = x

                if self.has_consts:
                    state[:, self.const_indices] = self.const_values

                state = self.gate_layers(state)

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
