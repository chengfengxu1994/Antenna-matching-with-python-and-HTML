"""
S-parameter network operations: port termination, interconnection, cascade.

Key operations for RF matching:
1. Port reduction: terminate a port with a known reflection coefficient
2. Port interconnection: connect a 2-port network between two ports of an N-port
3. Cascade: chain two 2-port networks in series
4. S ↔ Z ↔ Y conversion
"""

import numpy as np
from typing import Optional


def s_to_z(S: np.ndarray, Z0: float = 50.0) -> np.ndarray:
    """Convert S-matrix to Z-matrix (all ports have same Z0)."""
    n = S.shape[0]
    I = np.eye(n)
    Z0vec = Z0 * I
    # Z = Z0 * (I - S)^-1 * (I + S)
    Z = Z0vec @ np.linalg.solve(I - S, I + S)
    return Z


def z_to_s(Z: np.ndarray, Z0: float = 50.0) -> np.ndarray:
    """Convert Z-matrix to S-matrix (all ports have same Z0)."""
    n = Z.shape[0]
    I = np.eye(n)
    Z0vec = Z0 * I
    # S = (Z/Z0 - I) * (Z/Z0 + I)^-1
    Z_norm = Z / Z0
    S = np.linalg.solve(Z_norm + I, Z_norm - I)
    return S


def s_to_y(S: np.ndarray, Z0: float = 50.0) -> np.ndarray:
    """Convert S-matrix to Y-matrix."""
    Y0 = 1.0 / Z0
    n = S.shape[0]
    I = np.eye(n)
    # Y = Y0 * (I - S) * (I + S)^-1
    Y = Y0 * (I - S) @ np.linalg.inv(I + S)
    return Y


def y_to_s(Y: np.ndarray, Z0: float = 50.0) -> np.ndarray:
    """Convert Y-matrix to S-matrix."""
    Y0 = 1.0 / Z0
    n = Y.shape[0]
    I = np.eye(n)
    # S = (I - Y/Y0) * (I + Y/Y0)^-1
    Y_norm = Y / Y0
    S = (I - Y_norm) @ np.linalg.inv(I + Y_norm)
    return S


def terminate_port(S: np.ndarray, port: int, gamma: complex) -> np.ndarray:
    """
    Reduce an N-port S-matrix by terminating one port with reflection coefficient gamma.

    Args:
        S: NxN S-parameter matrix
        port: port index to terminate (0-based)
        gamma: reflection coefficient of the termination
               gamma = 1 for open, gamma = -1 for short, gamma = 0 for matched load

    Returns:
        (N-1)x(N-1) reduced S-matrix
    """
    n = S.shape[0]
    if n <= 1:
        raise ValueError("Cannot reduce a 1-port network further")

    # Formula: S'_ij = S_ij + (S_i,port * gamma * S_port,j) / (1 - gamma * S_port,port)
    new_n = n - 1
    S_new = np.zeros((new_n, new_n), dtype=complex)

    k = port  # terminated port index

    new_i = 0
    for i in range(n):
        if i == k:
            continue
        new_j = 0
        for j in range(n):
            if j == k:
                continue
            S_new[new_i, new_j] = S[i, j] + (S[i, k] * gamma * S[k, j]) / (1.0 - gamma * S[k, k])
            new_j += 1
        new_i += 1

    return S_new


def terminate_ports(S: np.ndarray, terminations: dict) -> np.ndarray:
    """
    Terminate multiple ports with given reflection coefficients.

    Args:
        S: NxN S-matrix
        terminations: dict mapping 0-based port index to gamma

    Returns:
        Reduced S-matrix with terminated ports removed
    """
    # Create a mapping from old indices to new indices
    n = S.shape[0]
    remaining = [i for i in range(n) if i not in terminations]

    if not remaining:
        raise ValueError("All ports terminated")

    # Use the general formula for multiple port terminations
    # Partition S-matrix into kept (a) and terminated (b) ports
    a_indices = remaining  # kept ports
    b_indices = list(terminations.keys())  # terminated ports

    S_aa = S[np.ix_(a_indices, a_indices)]
    S_ab = S[np.ix_(a_indices, b_indices)]
    S_ba = S[np.ix_(b_indices, a_indices)]
    S_bb = S[np.ix_(b_indices, b_indices)]

    # Gamma_b is diagonal matrix of termination reflection coefficients
    gamma_b = np.diag([terminations[idx] for idx in b_indices])

    # S_reduced = S_aa + S_ab * Gamma_b * (I - S_bb * Gamma_b)^-1 * S_ba
    nb = len(b_indices)
    I_b = np.eye(nb)
    inner = np.linalg.inv(I_b - S_bb @ gamma_b)
    S_reduced = S_aa + S_ab @ gamma_b @ inner @ S_ba

    return S_reduced


def cascade_2port(S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
    """
    Cascade two 2-port networks: [S1] → [S2]
    Uses T-parameter (transfer scattering) matrix multiplication.

    Args:
        S1: 2x2 S-matrix of first network
        S2: 2x2 S-matrix of second network

    Returns:
        2x2 S-matrix of cascaded network
    """
    # Convert S to T parameters
    T1 = s2p_to_t(S1)
    T2 = s2p_to_t(S2)

    # Cascade: T = T1 @ T2
    T = T1 @ T2

    # Convert back to S
    return t_to_s2p(T)


def s2p_to_t(S: np.ndarray) -> np.ndarray:
    """Convert 2x2 S-matrix to T-matrix (transfer scattering)."""
    # T = [[T11, T12], [T21, T22]]
    # T11 = -det(S)/S21, T12 = S11/S21
    # T21 = -S22/S21, T22 = 1/S21
    det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
    T = np.zeros((2, 2), dtype=complex)
    T[0, 0] = -det / S[1, 0]
    T[0, 1] = S[0, 0] / S[1, 0]
    T[1, 0] = -S[1, 1] / S[1, 0]
    T[1, 1] = 1.0 / S[1, 0]
    return T


def t_to_s2p(T: np.ndarray) -> np.ndarray:
    """Convert T-matrix back to 2x2 S-matrix."""
    S = np.zeros((2, 2), dtype=complex)
    S[0, 0] = T[0, 1] / T[1, 1]
    S[0, 1] = (T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0]) / T[1, 1]
    S[1, 0] = 1.0 / T[1, 1]
    S[1, 1] = -T[1, 0] / T[1, 1]
    return S


def connect_2port_to_multiport(S_big: np.ndarray, S_comp: np.ndarray,
                                port_a: int, port_b: int) -> np.ndarray:
    """
    Connect a 2-port component between two ports of an N-port network.

    The component's port1 connects to port_a, port2 connects to port_b.
    This reduces the total network by 2 ports (the component's ports are internal).

    Uses the general N-port connection formula via Y-parameters.

    Args:
        S_big: NxN S-matrix of the main network
        S_comp: 2x2 S-matrix of the component
        port_a: 0-based index of first connection port on main network
        port_b: 0-based index of second connection port on main network

    Returns:
        (N)x(N) S-matrix with component embedded (component ports absorbed)
        Actually returns N x N since we add 2 ports first then connect them internally.
    """
    n = S_big.shape[0]

    # Step 1: Embed component into larger (N+2) network
    # Build block-diagonal S-matrix: main network + component
    N_total = n + 2
    S_total = np.zeros((N_total, N_total), dtype=complex)
    S_total[:n, :n] = S_big
    S_total[n:, n:] = S_comp

    # Step 2: Convert to Y-parameters for easier interconnection
    Y_total = s_to_y(S_total)

    # Step 3: Connect ports: port_a ↔ component_port1 (n), port_b ↔ component_port2 (n+1)
    # The connection condition: V_a = V_comp1, V_b = V_comp2 (parallel connection)
    # Current: I_a' = I_a + I_comp1, I_b' = I_b + I_comp2

    # We reduce from N+2 to N ports by connecting:
    # - port n (component p1) to port_a
    # - port n+1 (component p2) to port_b

    # Use the general interconnection formula for Y-parameters
    # We connect port n to port_a and port n+1 to port_b

    # Create connection matrix: pairs of (port_p, port_q) to connect
    connections = [(n, port_a), (n + 1, port_b)]

    Y_reduced = connect_ports_y(Y_total, connections)

    # Step 4: Convert back to S-parameters
    S_result = y_to_s(Y_reduced)

    return S_result


def connect_ports_y(Y: np.ndarray, connections: list) -> np.ndarray:
    """
    Connect pairs of ports in a Y-matrix (parallel connection).

    When port p and port q are connected in parallel:
    - V_p = V_q (voltage equality)
    - I_p' = I_p + I_q (current addition)

    Args:
        Y: NxN Y-matrix
        connections: list of (p, q) port pairs to connect (0-based indices)

    Returns:
        Reduced Y-matrix with connected ports merged
    """
    n = Y.shape[0]
    # Ports that are being merged (the second of each pair gets absorbed into first)
    ports_to_remove = set(q for _, q in connections)
    remaining = [i for i in range(n) if i not in ports_to_remove]

    m = len(remaining)
    Y_new = np.zeros((m, m), dtype=complex)

    # Build mapping: new_index -> list of old indices merged into this port
    merged_groups = {i: [i] for i in remaining}
    for p, q in connections:
        # Find which group p belongs to
        for new_idx, group in merged_groups.items():
            if p in group:
                if q not in group:
                    group.append(q)
                break

    # Recompute: for each pair of new ports, sum the admittances
    for ni, gi in enumerate(remaining):
        for nj, gj in enumerate(remaining):
            total = 0j
            for oi in merged_groups[gi]:
                for oj in merged_groups[gj]:
                    total += Y[oi, oj]
            Y_new[ni, nj] = total

    return Y_new


def embed_series_component(S_dut: np.ndarray, S_comp: np.ndarray,
                           port: int, to_ground: bool = False) -> np.ndarray:
    """
    Connect a 2-port component in SERIES with a port of the DUT.

    For series connection:
    - Component port 1 is connected to DUT port
    - Component port 2 becomes the new external port (if not to_ground)
    - If to_ground: component port 2 is grounded (terminated with gamma=-1)

    This is equivalent to cascading the component with the port.

    Args:
        S_dut: NxN S-matrix of DUT
        S_comp: 2x2 S-matrix of series component
        port: which port of DUT to connect in series
        to_ground: if True, terminate comp port2 to ground (shunt config)

    Returns:
        NxN S-matrix with component connected (if not to_ground)
        or (N-1)x(N-1) if to_ground
    """
    n = S_dut.shape[0]

    # Method: expand DUT, cascade component on the specified port, then reduce
    # Step 1: Build N+2 network (DUT + component as separate blocks)
    N_total = n + 2
    S_total = np.zeros((N_total, N_total), dtype=complex)
    S_total[:n, :n] = S_dut
    S_total[n:, n:] = S_comp

    # Step 2: Convert to T-parameters for the affected path
    # For series connection: we need to cascade component with the port
    # The component goes between the port and the outside

    # Simpler approach: use the "series embedding" formula
    # For a series element at port k:
    # S'_ij = S_ij - (S_ik * (1 - S11_comp) * S_kj) / (1 - S_kk * S11_comp)
    # This is for when component port2 is terminated in Z0
    # Actually this is equivalent to: component in series, then port continues

    # Most general: Use connection matrix approach
    # Component port1 (n) connects to DUT port
    # Component port2 (n+1) becomes the new external port for that port
    # OR is terminated if to_ground

    if to_ground:
        # Shunt component: component in series to ground from the DUT port
        # Connect comp port1 to DUT port, comp port2 to ground (gamma = -1 for short)
        # But for shunt elements going to ground, it's a short to ground
        S_temp = connect_2port_to_multiport(S_dut, S_comp, port, port)
        # Actually this is for parallel connection between two DUT ports
        # For shunt to ground, we need a different approach
        return _embed_shunt_to_ground(S_dut, S_comp, port)
    else:
        # Series component: cascade
        # Replace S_kk region: component in series with port k
        return _embed_series_on_port(S_dut, S_comp, port)


def _embed_series_on_port(S_dut: np.ndarray, S_comp: np.ndarray, port: int) -> np.ndarray:
    """
    Embed a 2-port component in series with a specific port.
    The component's port1 faces the DUT, port2 faces the outside world.

    Uses the formula for cascading a 2-port onto one port of an N-port.
    Reference: "Multiport Network Analysis with Arbitrary Terminations"
    """
    n = S_dut.shape[0]
    # We treat this as: the component is an adapter on port k
    # The combined S-parameters can be derived via signal flow graph

    # Convert component S to its T-matrix
    # Component: port1→DUT_port, port2→external
    S11c, S12c = S_comp[0, 0], S_comp[0, 1]
    S21c, S22c = S_comp[1, 0], S_comp[1, 1]

    k = port
    S_new = np.zeros((n, n), dtype=complex)

    for i in range(n):
        for j in range(n):
            if i == k and j == k:
                # S'_kk = S22c + S21c * S_kk * S12c / (1 - S11c * S_kk)
                S_new[i, j] = S22c + (S21c * S_dut[i, j] * S12c) / (1.0 - S11c * S_dut[i, j])
            elif i == k:
                # S'_kj = S21c * S_kj / (1 - S11c * S_kk)
                S_new[i, j] = (S21c * S_dut[i, j]) / (1.0 - S11c * S_dut[k, k])
            elif j == k:
                # S'_ik = S_ik * S12c / (1 - S11c * S_kk)
                S_new[i, j] = (S_dut[i, j] * S12c) / (1.0 - S11c * S_dut[k, k])
            else:
                # S'_ij = S_ij + S_ik * S11c * S_kj / (1 - S11c * S_kk)
                S_new[i, j] = S_dut[i, j] + (S_dut[i, k] * S11c * S_dut[k, j]) / (1.0 - S11c * S_dut[k, k])

    return S_new


def _embed_shunt_to_ground(S_dut: np.ndarray, S_comp: np.ndarray, port: int) -> np.ndarray:
    """
    Embed a 2-port component as a shunt element from a DUT port to ground.
    Component port1 connects to DUT port, port2 is shorted to ground (gamma=-1).

    First embed in series on the port, then terminate the new outer port to ground.
    Wait - for shunt, the component is connected between the DUT port and ground.
    The component is in parallel (shunt), not series.

    For shunt configuration: component port1 to DUT port, port2 to ground.
    This is equivalent to terminating the component's port2 with gamma=-1,
    then connecting port1 to the DUT port.
    """
    # Step 1: Terminate component's port2 with gamma=-1 (short to ground)
    S_comp_terminated = terminate_port(S_comp, 1, -1.0)  # port2=1 (0-based), short

    # Now S_comp_terminated is a 1-port (just the reflection coefficient at port1)
    gamma_shunt = S_comp_terminated[0, 0]

    # Step 2: This shunt element is connected in PARALLEL at the DUT port
    # For a shunt element, the admittance adds: Y'_port = Y_DUT_port + Y_shunt
    # Y_shunt = Y0 * (1 - gamma_shunt) / (1 + gamma_shunt)
    Y0 = 1.0 / 50.0  # normalized
    Y_shunt = Y0 * (1.0 - gamma_shunt) / (1.0 + gamma_shunt)

    # Convert DUT S to Y at this port
    Y_dut = s_to_y(S_dut)
    Y_dut[port, port] += Y_shunt

    # Convert back to S
    return y_to_s(Y_dut)


def compute_s11_after_matching(S_dut: np.ndarray,
                                components: list,
                                topology: dict,
                                Z0: float = 50.0) -> complex:
    """
    Compute S11 after applying matching network to the DUT.

    Args:
        S_dut: NxN S-matrix of DUT at operating frequency
        components: list of (S_2x2, port_a, port_b, config) tuples
                    config: 'series', 'shunt', 'parallel'
        topology: describes how components connect to DUT ports
        Z0: reference impedance

    Returns:
        Complex S11 at port 0 (input port) after matching
    """
    S = S_dut.copy()

    for comp_s, port_a, port_b, config in components:
        if config == 'series':
            S = _embed_series_on_port(S, comp_s, port_a)
        elif config == 'shunt':
            S = _embed_shunt_to_ground(S, comp_s, port_a)
        elif config == 'parallel':
            # Parallel component between two ports
            S = connect_2port_to_multiport(S, comp_s, port_a, port_b)
        else:
            raise ValueError(f"Unknown config: {config}")

    return S[0, 0]
