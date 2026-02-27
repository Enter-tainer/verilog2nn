"""Topological sort and layer assignment for gate-level circuits."""

from collections import defaultdict

from verilog2nn.netlist_parser import Circuit, Gate


def topological_sort_layers(circuit: Circuit) -> list[list[Gate]]:
    """Sort gates topologically and assign to layers.

    Gates in the same layer have no dependencies between them and can be
    computed in parallel. Layer assignment is based on longest path from inputs.

    Args:
        circuit: Parsed circuit with gates and port info.

    Returns:
        List of layers, each containing gates that can execute in parallel.
    """
    # Build net -> producing gate mapping
    net_producer: dict[int, Gate] = {}
    for gate in circuit.gates:
        net_producer[gate.output_net] = gate

    # Primary inputs and constants are available at layer -1 (before layer 0)
    available_nets = set(circuit.input_bits) | set(circuit.const_nets.keys())

    # Build gate dependency graph
    gate_deps: dict[str, set[str]] = defaultdict(set)  # gate_name -> set of gate_names it depends on
    gate_by_name: dict[str, Gate] = {g.name: g for g in circuit.gates}

    for gate in circuit.gates:
        for inp_net in gate.input_nets:
            if inp_net in net_producer:
                gate_deps[gate.name].add(net_producer[inp_net].name)

    # Compute layer for each gate (longest path from inputs)
    gate_layer: dict[str, int] = {}

    def compute_layer(gate_name: str, visited: set[str]) -> int:
        if gate_name in gate_layer:
            return gate_layer[gate_name]

        if gate_name in visited:
            raise ValueError(f"Cycle detected involving gate {gate_name}")

        visited.add(gate_name)
        deps = gate_deps.get(gate_name, set())
        if not deps:
            gate_layer[gate_name] = 0
        else:
            gate_layer[gate_name] = (
                max(compute_layer(d, visited) for d in deps) + 1
            )
        visited.discard(gate_name)
        return gate_layer[gate_name]

    for gate in circuit.gates:
        compute_layer(gate.name, set())

    # Group gates by layer
    max_layer = max(gate_layer.values()) if gate_layer else -1
    layers: list[list[Gate]] = [[] for _ in range(max_layer + 1)]
    for gate in circuit.gates:
        layer_idx = gate_layer[gate.name]
        layers[layer_idx].append(gate)

    return layers
