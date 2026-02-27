"""Parse Yosys JSON netlist into a gate-level circuit representation."""

from dataclasses import dataclass, field


@dataclass
class Gate:
    """A single logic gate in the netlist."""

    name: str
    gate_type: str  # AND, NOT
    input_nets: list[int]  # net IDs for inputs
    output_net: int  # net ID for output


@dataclass
class Circuit:
    """Parsed gate-level circuit."""

    input_bits: list[int]  # net IDs for primary inputs (ordered)
    output_bits: list[int]  # net IDs for primary outputs (ordered)
    input_names: list[str]  # port names, one per bit (e.g., "a[0]", "a[1]")
    output_names: list[str]  # port names, one per bit
    gates: list[Gate] = field(default_factory=list)
    const_nets: dict[int, int] = field(default_factory=dict)  # net_id -> 0 or 1


# Yosys constant bit markers
CONST_0 = "0"
CONST_1 = "1"
CONST_X = "x"
CONST_Z = "z"


def _expand_port_bits(port_name: str, bits: list) -> list[tuple[str, int | str]]:
    """Expand a port into (name, bit_id) pairs, handling multi-bit signals."""
    if len(bits) == 1:
        return [(port_name, bits[0])]
    return [(f"{port_name}[{i}]", b) for i, b in enumerate(bits)]


def parse_netlist(netlist_json: dict, top: str | None = None) -> Circuit:
    """Parse Yosys JSON netlist into Circuit.

    Args:
        netlist_json: Parsed JSON dict from Yosys write_json output.
        top: Top module name. If None, use the first (or only) module.

    Returns:
        Circuit with gates, input/output ports.
    """
    modules = netlist_json.get("modules", {})
    if not modules:
        raise ValueError("No modules found in netlist JSON")

    if top:
        if top not in modules:
            raise ValueError(f"Module '{top}' not found. Available: {list(modules)}")
        module = modules[top]
    else:
        # Use the module with top attribute, or the first one
        for name, mod in modules.items():
            attrs = mod.get("attributes", {})
            if attrs.get("top") == "00000000000000000000000000000001":
                top = name
                module = mod
                break
        else:
            top = next(iter(modules))
            module = modules[top]

    ports = module.get("ports", {})
    cells = module.get("cells", {})

    # Collect input and output port bits
    input_bits = []
    output_bits = []
    input_names = []
    output_names = []

    for port_name, port_info in ports.items():
        direction = port_info["direction"]
        bits = port_info["bits"]

        expanded = _expand_port_bits(port_name, bits)
        for bit_name, bit_id in expanded:
            if direction == "input":
                input_names.append(bit_name)
                input_bits.append(bit_id)
            elif direction == "output":
                output_names.append(bit_name)
                output_bits.append(bit_id)

    # Track constant nets
    const_nets: dict[int, int] = {}
    # We need a synthetic net ID for constants since they use string markers
    next_const_id = 1_000_000

    def resolve_bit(b) -> int:
        """Resolve a bit reference to a net ID, handling constants."""
        nonlocal next_const_id
        if isinstance(b, int):
            return b
        if b == CONST_0:
            cid = next_const_id
            next_const_id += 1
            const_nets[cid] = 0
            return cid
        if b == CONST_1:
            cid = next_const_id
            next_const_id += 1
            const_nets[cid] = 1
            return cid
        # x or z treated as 0
        cid = next_const_id
        next_const_id += 1
        const_nets[cid] = 0
        return cid

    # Also resolve output bits that might be constants
    resolved_output_bits = []
    for b in output_bits:
        if isinstance(b, str):
            resolved = resolve_bit(b)
            resolved_output_bits.append(resolved)
        else:
            resolved_output_bits.append(b)
    output_bits = resolved_output_bits

    # Parse cells (gates)
    gates = []
    gate_type_map = {
        "$_AND_": "AND",
        "$_NOT_": "NOT",
    }

    # Cell types to skip (Yosys metadata, not actual gates)
    skip_types = {"$scopeinfo"}

    for cell_name, cell_info in cells.items():
        cell_type = cell_info["type"]
        if cell_type in skip_types:
            continue
        if cell_type not in gate_type_map:
            raise ValueError(
                f"Unsupported cell type: {cell_type}. "
                f"Expected one of {list(gate_type_map)}. "
                f"Make sure Yosys maps to AND/NOT gates only (abc -g AND)."
            )

        gate_type = gate_type_map[cell_type]
        connections = cell_info["connections"]

        if gate_type == "NOT":
            inp = [resolve_bit(connections["A"][0])]
        else:
            inp = [
                resolve_bit(connections["A"][0]),
                resolve_bit(connections["B"][0]),
            ]

        out = resolve_bit(connections["Y"][0])
        gates.append(Gate(name=cell_name, gate_type=gate_type, input_nets=inp, output_net=out))

    return Circuit(
        input_bits=input_bits,
        output_bits=output_bits,
        input_names=input_names,
        output_names=output_names,
        gates=gates,
        const_nets=const_nets,
    )
