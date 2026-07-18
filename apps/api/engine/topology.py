"""
RF Matching Topology definitions.

A topology specifies how matching components are connected to the DUT ports.
Supported topologies:
- L-network: 2 components (series-L + shunt-C or series-C + shunt-L)
- Pi-network: 3 components (shunt-series-shunt)
- T-network: 3 components (series-shunt-series)
- Ladder-N: N components alternating series/shunt
- Custom: user-defined connections

Each topology element has:
- position: which component in the chain
- connection_type: 'series' or 'shunt'
- port: which DUT port it connects to (0-based)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


class ConnectionType(Enum):
    SERIES = 'series'      # Component in series with a port
    SHUNT = 'shunt'        # Component from port to ground
    PARALLEL = 'parallel'  # Component between two ports


@dataclass
class TopologyElement:
    """One element in a matching topology."""
    position: int           # Position in chain (0-based)
    connection_type: ConnectionType
    port: int               # DUT port this element connects to (0-based)
    port2: Optional[int] = None  # For PARALLEL: second port
    component_type: str = 'any'  # 'inductor', 'capacitor', or 'any'


@dataclass
class Topology:
    """A matching topology definition."""
    name: str
    num_components: int
    elements: List[TopologyElement] = field(default_factory=list)
    description: str = ""

    @property
    def n_components(self):
        return self.num_components

    def remap_ports(self, old_port: int = 0, new_port: int = 0) -> "Topology":
        """Create a copy with all element ports remapped. No-op if old_port == new_port."""
        if old_port == new_port:
            return self
        new_elements = []
        for elem in self.elements:
            new_elements.append(TopologyElement(
                position=elem.position,
                connection_type=elem.connection_type,
                port=new_port if elem.port == old_port else elem.port,
                port2=new_port if elem.port2 == old_port else elem.port2,
                component_type=elem.component_type,
            ))
        return Topology(
            name=self.name,
            num_components=self.num_components,
            elements=new_elements,
            description=self.description,
        )


# ─── Standard Topology Library ───

def _make_topology(name: str, desc: str, elements_spec: List[dict]) -> Topology:
    """Helper to create a topology from a spec list."""
    elements = []
    for i, spec in enumerate(elements_spec):
        conn_type_str = spec.get('type', 'series').lower()
        if conn_type_str == 'series':
            ct = ConnectionType.SERIES
        elif conn_type_str == 'shunt':
            ct = ConnectionType.SHUNT
        elif conn_type_str == 'parallel':
            ct = ConnectionType.PARALLEL
        else:
            ct = ConnectionType.SERIES

        elem = TopologyElement(
            position=i,
            connection_type=ct,
            port=spec.get('port', 0),
            port2=spec.get('port2'),
            component_type=spec.get('comp_type', 'any')
        )
        elements.append(elem)

    return Topology(
        name=name,
        num_components=len(elements),
        elements=elements,
        description=desc
    )


def get_standard_topologies() -> List[Topology]:
    """Get the standard matching topology library."""
    topologies = []

    # ─── 1-component networks ───
    # Useful both for quick matching and for explicit manual fixture tuning.
    for connection in ('series', 'shunt'):
        for component_type, symbol in (('inductor', 'L'), ('capacitor', 'C')):
            label = 'Series' if connection == 'series' else 'Shunt'
            topologies.append(_make_topology(
                f"1-Element ({label}-{symbol})",
                f"Single {connection} {component_type} at port 0",
                [{'type': connection, 'port': 0, 'comp_type': component_type}],
            ))

    # ─── 2-component L-Networks ───
    # L-network: port-based (all components on port 0, the input port)
    # Variant A: series-L then shunt-C
    topologies.append(_make_topology(
        "L-Network (Series-L, Shunt-C)",
        "Series inductor at port 0, shunt capacitor to ground at port 0",
        [
            {'type': 'series', 'port': 0, 'comp_type': 'inductor'},
            {'type': 'shunt', 'port': 0, 'comp_type': 'capacitor'},
        ]
    ))
    # Variant B: series-C then shunt-L
    topologies.append(_make_topology(
        "L-Network (Series-C, Shunt-L)",
        "Series capacitor at port 0, shunt inductor to ground at port 0",
        [
            {'type': 'series', 'port': 0, 'comp_type': 'capacitor'},
            {'type': 'shunt', 'port': 0, 'comp_type': 'inductor'},
        ]
    ))
    # Variant C: shunt-L then series-C
    topologies.append(_make_topology(
        "L-Network (Shunt-L, Series-C)",
        "Shunt inductor at port 0, series capacitor at port 0",
        [
            {'type': 'shunt', 'port': 0, 'comp_type': 'inductor'},
            {'type': 'series', 'port': 0, 'comp_type': 'capacitor'},
        ]
    ))
    # Variant D: shunt-C then series-L
    topologies.append(_make_topology(
        "L-Network (Shunt-C, Series-L)",
        "Shunt capacitor at port 0, series inductor at port 0",
        [
            {'type': 'shunt', 'port': 0, 'comp_type': 'capacitor'},
            {'type': 'series', 'port': 0, 'comp_type': 'inductor'},
        ]
    ))

    # ─── 3-component Pi-Networks ───
    existing_two_elem = {
        tuple((e.connection_type.value, e.component_type) for e in t.elements)
        for t in topologies
        if t.num_components == 2
    }
    conn_labels = {'series': 'Series', 'shunt': 'Shunt'}
    type_labels = {'inductor': 'L', 'capacitor': 'C'}
    for comp_a in ['inductor', 'capacitor']:
        for comp_b in ['inductor', 'capacitor']:
            for conn_a in ['series', 'shunt']:
                for conn_b in ['series', 'shunt']:
                    signature = ((conn_a, comp_a), (conn_b, comp_b))
                    if signature in existing_two_elem:
                        continue
                    topologies.append(_make_topology(
                        f"2-Element ({conn_labels[conn_a]}-{type_labels[comp_a]}, "
                        f"{conn_labels[conn_b]}-{type_labels[comp_b]})",
                        "Complete two-element search topology",
                        [
                            {'type': conn_a, 'port': 0, 'comp_type': comp_a},
                            {'type': conn_b, 'port': 0, 'comp_type': comp_b},
                        ]
                    ))
                    existing_two_elem.add(signature)

    # Pi: shunt - series - shunt (all on port 0)
    topologies.append(_make_topology(
        "Pi-Network (C-L-C)",
        "Shunt C, series L, shunt C at port 0",
        [
            {'type': 'shunt', 'port': 0, 'comp_type': 'capacitor'},
            {'type': 'series', 'port': 0, 'comp_type': 'inductor'},
            {'type': 'shunt', 'port': 0, 'comp_type': 'capacitor'},
        ]
    ))
    topologies.append(_make_topology(
        "Pi-Network (L-C-L)",
        "Shunt L, series C, shunt L at port 0",
        [
            {'type': 'shunt', 'port': 0, 'comp_type': 'inductor'},
            {'type': 'series', 'port': 0, 'comp_type': 'capacitor'},
            {'type': 'shunt', 'port': 0, 'comp_type': 'inductor'},
        ]
    ))

    # ─── 3-component T-Networks ───
    # T: series - shunt - series
    topologies.append(_make_topology(
        "T-Network (L-C-L)",
        "Series L, shunt C, series L at port 0",
        [
            {'type': 'series', 'port': 0, 'comp_type': 'inductor'},
            {'type': 'shunt', 'port': 0, 'comp_type': 'capacitor'},
            {'type': 'series', 'port': 0, 'comp_type': 'inductor'},
        ]
    ))
    topologies.append(_make_topology(
        "T-Network (C-L-C)",
        "Series C, shunt L, series C at port 0",
        [
            {'type': 'series', 'port': 0, 'comp_type': 'capacitor'},
            {'type': 'shunt', 'port': 0, 'comp_type': 'inductor'},
            {'type': 'series', 'port': 0, 'comp_type': 'capacitor'},
        ]
    ))

    # ─── 4-component Ladder (L-C-L-C) ───
    topologies.append(_make_topology(
        "4-Element Ladder (Series-L, Shunt-C, Series-L, Shunt-C)",
        "Ladder network: L series, C shunt, L series, C shunt",
        [
            {'type': 'series', 'port': 0, 'comp_type': 'inductor'},
            {'type': 'shunt', 'port': 0, 'comp_type': 'capacitor'},
            {'type': 'series', 'port': 0, 'comp_type': 'inductor'},
            {'type': 'shunt', 'port': 0, 'comp_type': 'capacitor'},
        ]
    ))
    topologies.append(_make_topology(
        "4-Element Ladder (Series-C, Shunt-L, Series-C, Shunt-L)",
        "Ladder network: C series, L shunt, C series, L shunt",
        [
            {'type': 'series', 'port': 0, 'comp_type': 'capacitor'},
            {'type': 'shunt', 'port': 0, 'comp_type': 'inductor'},
            {'type': 'series', 'port': 0, 'comp_type': 'capacitor'},
            {'type': 'shunt', 'port': 0, 'comp_type': 'inductor'},
        ]
    ))

    return topologies


def generate_ladder_topologies(num_components: int) -> List[Topology]:
    """
    Generate all ladder topologies with N components.
    Each position can be series or shunt, with inductor or capacitor.
    This generates 2^N configurations (each can be series or shunt)
    × 2^N component type assignments (L or C).

    Total: 4^N possible topologies. For N=5: 1024 topologies.
    But we filter to alternating patterns (series-shunt-series... or shunt-series-shunt...).
    """
    topologies = []

    # Alternating patterns only (most practical)
    for start_type in [ConnectionType.SERIES, ConnectionType.SHUNT]:
        for lc_pattern in range(2 ** num_components):
            elements = []
            for i in range(num_components):
                ct = ConnectionType.SERIES if ((i % 2) == (0 if start_type == ConnectionType.SERIES else 1)) else ConnectionType.SHUNT
                comp_type = 'inductor' if ((lc_pattern >> i) & 1) else 'capacitor'
                elements.append({
                    'type': 'series' if ct == ConnectionType.SERIES else 'shunt',
                    'port': 0,
                    'comp_type': comp_type,
                })

            name = f"Ladder-{num_components} ({start_type.name}-first)"
            topologies.append(_make_topology(name, f"{num_components}-element ladder", elements))

    return topologies


def topology_to_components_list(topology: Topology) -> List[dict]:
    """
    Convert a topology to a list of component configurations
    that can be passed to the network engine.
    """
    return [
        {
            'position': elem.position,
            'connection_type': elem.connection_type.value,
            'port': elem.port,
            'port2': elem.port2,
            'component_type': elem.component_type,
        }
        for elem in topology.elements
    ]
