from __future__ import annotations

import numpy as np

from .models import Element


def flip_s2p_ports(s_parameters: np.ndarray) -> np.ndarray:
    """Swap ports 1 and 2 of one S2P matrix or a frequency sweep."""
    data = np.asarray(s_parameters, dtype=complex)
    if data.shape[-2:] != (2, 2):
        raise ValueError("port flipping requires S-parameters ending in shape (2, 2)")
    return data[..., ::-1, ::-1].copy()


def renormalize_s_parameters(
    s_parameters: np.ndarray,
    source_z0: float | np.ndarray,
    target_z0: float | np.ndarray,
) -> np.ndarray:
    """Renormalize real scalar/per-port power-wave S parameters directly in wave space."""
    data = np.asarray(s_parameters, dtype=complex)
    if data.ndim < 2 or data.shape[-1] != data.shape[-2]:
        raise ValueError("S-parameters must end in a square port matrix")
    port_count = data.shape[-1]
    source_root = _z0_matrix(source_z0, port_count)
    target_root = _z0_matrix(target_z0, port_count)
    if np.array_equal(source_root, target_root):
        return data.copy()
    source_inverse = np.linalg.inv(source_root)
    target_inverse = np.linalg.inv(target_root)
    p = 0.5 * (
        source_inverse @ target_root + source_root @ target_inverse
    )
    q = 0.5 * (
        source_inverse @ target_root - source_root @ target_inverse
    )
    return np.linalg.solve(p - data @ q, data @ p - q)


def s2p_to_chain(s_parameters: np.ndarray, minimum_transmission: float = 1e-12) -> np.ndarray:
    """Convert S2P to a transfer-scattering chain matrix, with an invertibility guard."""
    data = np.asarray(s_parameters, dtype=complex)
    if data.shape[-2:] != (2, 2):
        raise ValueError("chain conversion requires S-parameters ending in shape (2, 2)")
    s21 = data[..., 1, 0]
    if np.any(np.abs(s21) < minimum_transmission):
        raise ValueError("S2P is not chain-convertible because |S21| is too small")
    out = np.empty_like(data)
    determinant = data[..., 0, 0] * data[..., 1, 1] - data[..., 0, 1] * s21
    out[..., 0, 0] = -determinant / s21
    out[..., 0, 1] = data[..., 0, 0] / s21
    out[..., 1, 0] = -data[..., 1, 1] / s21
    out[..., 1, 1] = 1.0 / s21
    return out


def chain_to_s2p(chain_parameters: np.ndarray, minimum_denominator: float = 1e-12) -> np.ndarray:
    data = np.asarray(chain_parameters, dtype=complex)
    if data.shape[-2:] != (2, 2):
        raise ValueError("chain parameters must end in shape (2, 2)")
    denominator = data[..., 1, 1]
    if np.any(np.abs(denominator) < minimum_denominator):
        raise ValueError("chain matrix cannot be converted to S2P because |T22| is too small")
    out = np.empty_like(data)
    out[..., 0, 0] = data[..., 0, 1] / denominator
    out[..., 0, 1] = (
        data[..., 0, 0] * denominator - data[..., 0, 1] * data[..., 1, 0]
    ) / denominator
    out[..., 1, 0] = 1.0 / denominator
    out[..., 1, 1] = -data[..., 1, 0] / denominator
    return out


def cascade_s2p(*networks: np.ndarray) -> np.ndarray:
    if not networks:
        raise ValueError("at least one S2P network is required")
    chain = s2p_to_chain(networks[0])
    for network in networks[1:]:
        chain = chain @ s2p_to_chain(network)
    return chain_to_s2p(chain)


def deembed_s2p(
    measured: np.ndarray,
    *,
    left_fixture: np.ndarray | None = None,
    right_fixture: np.ndarray | None = None,
    maximum_condition_number: float = 1e10,
) -> tuple[np.ndarray, dict]:
    """Recover DUT from measured = left fixture → DUT → right fixture."""
    measured_chain = s2p_to_chain(measured)
    recovered = measured_chain.copy()
    conditions = {}
    if left_fixture is not None:
        left_chain = s2p_to_chain(left_fixture)
        conditions["left"] = float(np.max(np.linalg.cond(left_chain)))
        if conditions["left"] > maximum_condition_number:
            raise ValueError(f"left fixture chain matrix is ill-conditioned ({conditions['left']:.6g})")
        recovered = np.linalg.solve(left_chain, recovered)
    if right_fixture is not None:
        right_chain = s2p_to_chain(right_fixture)
        conditions["right"] = float(np.max(np.linalg.cond(right_chain)))
        if conditions["right"] > maximum_condition_number:
            raise ValueError(f"right fixture chain matrix is ill-conditioned ({conditions['right']:.6g})")
        recovered = np.swapaxes(np.linalg.solve(
            np.swapaxes(right_chain, -1, -2), np.swapaxes(recovered, -1, -2)
        ), -1, -2)
    result = chain_to_s2p(recovered)
    parts = [part for part in (left_fixture, result, right_fixture) if part is not None]
    residual = float(np.max(np.abs(cascade_s2p(*parts) - measured)))
    return result, {
        "maximum_fixture_condition_number": max(conditions.values(), default=1.0),
        "fixture_condition_numbers": conditions,
        "maximum_recascade_residual": residual,
    }


def _z0_matrix(z0: float | np.ndarray, n: int) -> np.ndarray:
    values = np.full(n, float(z0)) if np.ndim(z0) == 0 else np.asarray(z0, dtype=float)
    if values.shape != (n,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("z0 must be finite positive scalar or one value per port")
    return np.diag(np.sqrt(values))


def s_to_z(s: np.ndarray, z0: float | np.ndarray = 50.0) -> np.ndarray:
    s = np.asarray(s, dtype=complex)
    n = s.shape[-1]
    root = _z0_matrix(z0, n)
    eye = np.eye(n, dtype=complex)
    rhs = np.broadcast_to(root, s.shape) if s.ndim > 2 else root
    return root @ (eye + s) @ np.linalg.solve(eye - s, rhs)


def z_to_s(z: np.ndarray, z0: float | np.ndarray = 50.0) -> np.ndarray:
    z = np.asarray(z, dtype=complex)
    n = z.shape[-1]
    root = _z0_matrix(z0, n)
    normalized = np.linalg.inv(root) @ z @ np.linalg.inv(root)
    eye = np.eye(n, dtype=complex)
    solved = np.linalg.solve(
        np.swapaxes(normalized + eye, -1, -2),
        np.swapaxes(normalized - eye, -1, -2),
    )
    return np.swapaxes(solved, -1, -2)


def terminate(s: np.ndarray, terminations: dict[int, complex]) -> tuple[np.ndarray, list[int]]:
    """Terminate ports and return reduced S plus its original-port ordering."""
    n = s.shape[0]
    terminated = sorted(terminations)
    kept = [p for p in range(n) if p not in terminations]
    if not terminated:
        return s.copy(), kept
    see = s[np.ix_(kept, kept)]
    sei = s[np.ix_(kept, terminated)]
    sie = s[np.ix_(terminated, kept)]
    sii = s[np.ix_(terminated, terminated)]
    gamma = np.diag([terminations[p] for p in terminated])
    correction = sei @ np.linalg.solve(np.eye(len(terminated)) - gamma @ sii, gamma @ sie)
    return see + correction, kept


def element_impedance(element: Element, frequency_hz: float) -> complex:
    omega = 2.0 * np.pi * frequency_hz
    if element.kind == "L":
        return 1j * omega * element.value
    return 1.0 / (1j * omega * element.value)


def apply_elements(s: np.ndarray, elements: list[Element], frequency_hz: float, z0: float | np.ndarray = 50.0) -> np.ndarray:
    """Apply ideal lumped elements in Z/Y form without reducing port count."""
    out = np.asarray(s, dtype=complex)
    n = out.shape[0]
    for element in elements:
        if not 0 <= element.port < n:
            raise IndexError(f"element port {element.port} outside {n}-port network")
        impedance = element_impedance(element, frequency_hz)
        z = s_to_z(out, z0)
        if element.connection == "series":
            z[element.port, element.port] += impedance
        else:
            y = np.linalg.inv(z)
            y[element.port, element.port] += 1.0 / impedance
            z = np.linalg.inv(y)
        out = z_to_s(z, z0)
    return out


def apply_elements_sweep(
    s_parameters: np.ndarray,
    elements: list[Element],
    frequencies_hz: np.ndarray,
    z0: float | np.ndarray = 50.0,
) -> np.ndarray:
    """Apply one ideal network to every frequency using batched matrix algebra."""
    s_parameters = np.asarray(s_parameters, dtype=complex)
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    if s_parameters.ndim != 3 or s_parameters.shape[0] != len(frequencies_hz) or s_parameters.shape[1] != s_parameters.shape[2]:
        raise ValueError("s_parameters must have shape (frequency, port, port)")
    if np.any(frequencies_hz <= 0):
        raise ValueError("frequencies must be positive")
    n_ports = s_parameters.shape[1]
    z = s_to_z(s_parameters, z0)
    omega = 2.0 * np.pi * frequencies_hz
    for element in elements:
        if not 0 <= element.port < n_ports:
            raise IndexError(f"element port {element.port} outside {n_ports}-port network")
        impedance = 1j * omega * element.value if element.kind == "L" else 1.0 / (1j * omega * element.value)
        if element.connection == "series":
            z[:, element.port, element.port] += impedance
        else:
            y = np.linalg.inv(z)
            y[:, element.port, element.port] += 1.0 / impedance
            z = np.linalg.inv(y)
    return z_to_s(z, z0)
