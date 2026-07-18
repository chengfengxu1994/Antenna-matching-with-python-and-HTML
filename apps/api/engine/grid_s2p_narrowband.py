"""
Narrow-band S2P grid optimizer.

This engine is intended for difficult narrow-band cases where the generic
candidate/beam optimizer can miss a dual-band solution during phase 1.  It
interpolates both DUT SNP and component S2P data to the requested band points,
then evaluates real component S-parameters directly.

The reported score is matching/accepted-power quality, not measured antenna
radiation efficiency.  Callers should label it accordingly unless radiation
efficiency data is supplied elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .touchstone import TouchstoneData, load_touchstone_file
from project_paths import MURATA_DIR


Z0 = 50.0
DEFAULT_DB_PATH = MURATA_DIR / "optenni_components.db"


@dataclass
class GridComponent:
    part_number: str
    component_type: str
    nominal_value: float
    nominal_unit: str
    s2p_path: str

    def get_s_matrix_at_freq(self, freq_hz: float) -> np.ndarray:
        data = load_touchstone_file(self.s2p_path)
        return data.get_s_matrix_interpolated(freq_hz)


@dataclass
class GridChoice:
    position: int
    component: GridComponent
    connection_type: str
    port: int
    port2: Optional[int] = None


def _band_points_hz(bands_mhz: Sequence[Sequence[float]], points_per_band: int) -> np.ndarray:
    freqs = []
    n = max(1, int(points_per_band))
    for band in bands_mhz:
        start, stop = float(band[0]), float(band[1])
        if abs(stop - start) < 1e-12 or n == 1:
            freqs.append(0.5 * (start + stop) * 1e6)
        else:
            freqs.extend(np.linspace(start * 1e6, stop * 1e6, n).tolist())
    return np.array(sorted(set(float(f) for f in freqs)), dtype=float)


def _s_to_y(s_matrix: np.ndarray) -> np.ndarray:
    ident = np.eye(2, dtype=complex)
    out = np.empty_like(s_matrix)
    for idx, s in enumerate(s_matrix):
        out[idx] = (1.0 / Z0) * (ident - s) @ np.linalg.pinv(ident + s)
    return out


def _series_s(comp_data: dict) -> np.ndarray:
    return comp_data["s"]


def _shunt_s(comp_data: dict) -> np.ndarray:
    ys = comp_data["y"][:, 0, 0]
    y0 = 1.0 / Z0
    denom = 2.0 * y0 + ys
    denom = np.where(np.abs(denom) < 1e-15, 1e-15 + 0j, denom)
    s = np.zeros((len(ys), 2, 2), dtype=complex)
    s[:, 0, 0] = -ys / denom
    s[:, 1, 1] = -ys / denom
    s[:, 0, 1] = 2.0 * y0 / denom
    s[:, 1, 0] = 2.0 * y0 / denom
    return s


def _cascade_2port(sa: np.ndarray, sb: np.ndarray) -> np.ndarray:
    denom = 1.0 / (1.0 - sa[:, 1, 1] * sb[:, 0, 0] + 1e-50)
    s11 = sa[:, 0, 0] + sa[:, 0, 1] * sb[:, 0, 0] * sa[:, 1, 0] * denom
    s21 = sa[:, 1, 0] * sb[:, 1, 0] * denom
    s12 = sa[:, 0, 1] * sb[:, 0, 1] * denom
    s22 = sb[:, 1, 1] + sb[:, 0, 1] * sa[:, 1, 1] * sb[:, 1, 0] * denom
    out = np.empty_like(sa)
    out[:, 0, 0] = s11
    out[:, 0, 1] = s12
    out[:, 1, 0] = s21
    out[:, 1, 1] = s22
    return out


def _cascade_network(parts: Sequence[np.ndarray], nf: int) -> np.ndarray:
    out = np.zeros((nf, 2, 2), dtype=complex)
    out[:, 0, 1] = 1.0
    out[:, 1, 0] = 1.0
    for part in parts:
        out = _cascade_2port(out, part)
    return out


def _transducer_gain(match_s: np.ndarray, load_gamma: np.ndarray) -> np.ndarray:
    return (
        np.abs(match_s[:, 1, 0]) ** 2
        * np.maximum(0.0, 1.0 - np.abs(load_gamma) ** 2)
        / (np.abs(1.0 - match_s[:, 1, 1] * load_gamma) ** 2 + 1e-50)
    )


def _input_gamma(match_s: np.ndarray, load_gamma: np.ndarray) -> np.ndarray:
    denom = 1.0 - match_s[:, 1, 1] * load_gamma
    denom = np.where(np.abs(denom) < 1e-15, 1e-15 + 0j, denom)
    return match_s[:, 0, 0] + match_s[:, 0, 1] * match_s[:, 1, 0] * load_gamma / denom


def _port_reflection(match_s: np.ndarray, port_type: str, ground_term: str = "short") -> np.ndarray:
    if port_type == "load":
        return match_s[:, 1, 1].copy()
    term = {"short": -1.0, "open": 1.0, "load": 0.0}.get(ground_term, -1.0)
    s11, s12 = match_s[:, 0, 0], match_s[:, 0, 1]
    s21, s22 = match_s[:, 1, 0], match_s[:, 1, 1]
    denom = 1.0 - s11 * term
    denom = np.where(np.abs(denom) < 1e-15, 1e-15 + 0j, denom)
    return s22 + s12 * s21 * term / denom


def _effective_1port(s_nport: np.ndarray, port_idx: int, gamma_dict: Dict[int, np.ndarray]) -> np.ndarray:
    n_ports = s_nport.shape[-1]
    nf = s_nport.shape[0]
    if n_ports == 1:
        return s_nport[:, 0, 0]
    if n_ports == 2:
        other = 1 - port_idx
        gamma = gamma_dict.get(other, np.zeros(nf, dtype=complex))
        if port_idx == 0:
            sii, sjj, sij, sji = s_nport[:, 0, 0], s_nport[:, 1, 1], s_nport[:, 0, 1], s_nport[:, 1, 0]
        else:
            sii, sjj, sij, sji = s_nport[:, 1, 1], s_nport[:, 0, 0], s_nport[:, 1, 0], s_nport[:, 0, 1]
        denom = 1.0 - sjj * gamma
        denom = np.where(np.abs(denom) < 1e-15, 1e-15 + 0j, denom)
        return sii + sij * sji * gamma / denom

    others = [p for p in range(n_ports) if p != port_idx]
    out = np.empty(nf, dtype=complex)
    ident = np.eye(len(others), dtype=complex)
    for fidx in range(nf):
        gamma = np.diag([gamma_dict.get(p, np.zeros(nf, dtype=complex))[fidx] for p in others])
        skk = s_nport[fidx, port_idx, port_idx]
        skt = s_nport[fidx, port_idx, others][None, :]
        stk = s_nport[fidx, others, port_idx][:, None]
        stt = s_nport[fidx][np.ix_(others, others)]
        x = np.linalg.pinv(ident - gamma @ stt) @ gamma @ stk
        out[fidx] = skk + (skt @ x)[0, 0]
    return out


def _vector_score(gt: np.ndarray) -> float:
    return float(np.mean(gt))


class NarrowbandGridS2P:
    def __init__(self, l_library, c_library, freqs_hz: np.ndarray):
        self.freqs_hz = np.asarray(freqs_hz, dtype=float)
        self.nf = len(self.freqs_hz)
        self.l_components = l_library
        self.c_components = c_library

    def _arr(self, comp_type: str, topo: str) -> np.ndarray:
        comps = self.l_components if comp_type == "L" else self.c_components
        pos = 2 if topo == "S" else 3
        return np.stack([c[pos] for c in comps])

    def _meta(self, comp_type: str):
        return self.l_components if comp_type == "L" else self.c_components

    def _best_for_port(self, topology: Tuple[str, ...], types: Tuple[str, ...],
                       load_gamma: np.ndarray, freq_mask: Optional[np.ndarray] = None):
        arrays = [self._arr(ct, tt) for tt, ct in zip(topology, types)]
        if freq_mask is None:
            freq_mask = np.ones(self.nf, dtype=bool)
        if len(arrays) == 1:
            scores = []
            for sm in arrays[0]:
                scores.append(_vector_score(_transducer_gain(sm, load_gamma)[freq_mask]))
            best = [int(np.argmax(scores))]
            score = float(max(scores))
        elif len(arrays) == 2:
            best, score = self._grid_2d(arrays, load_gamma, freq_mask)
        elif len(arrays) == 3:
            best, score = self._grid_3d(arrays, load_gamma, freq_mask)
        else:
            best, score = self._grid_nd_coarse(arrays, load_gamma, freq_mask)

        s_list = []
        values = []
        parts = []
        components = []
        for idx, tt, ct in zip(best, topology, types):
            meta = self._meta(ct)[idx]
            s_matrix = meta[2] if tt == "S" else meta[3]
            s_list.append(s_matrix)
            values.append(meta[0].nominal_value)
            parts.append(meta[0].part_number)
            components.append(meta[0])
        return {
            "topology": list(topology),
            "types": ["inductor" if t == "L" else "capacitor" for t in types],
            "type_codes": list(types),
            "values": values,
            "part_numbers": parts,
            "components": components,
            "s_list": s_list,
            "score": score,
        }

    def _grid_2d(self, arrays: Sequence[np.ndarray], load_gamma: np.ndarray, freq_mask: np.ndarray):
        a0, a1 = arrays
        denom = 1.0 / (1.0 - a0[:, None, :, 1, 1] * a1[None, :, :, 0, 0] + 1e-50)
        s21 = a0[:, None, :, 1, 0] * a1[None, :, :, 1, 0] * denom
        s12 = a0[:, None, :, 0, 1] * a1[None, :, :, 0, 1] * denom
        s22 = a1[None, :, :, 1, 1] + a1[None, :, :, 0, 1] * a0[:, None, :, 1, 1] * a1[None, :, :, 1, 0] * denom
        gt = (
            np.abs(s21) ** 2
            * np.maximum(0.0, 1.0 - np.abs(load_gamma) ** 2)
            / (np.abs(1.0 - s22 * load_gamma) ** 2 + 1e-50)
        )
        scores = np.mean(gt[..., freq_mask], axis=-1)
        best = np.unravel_index(int(np.argmax(scores)), scores.shape)
        return [int(best[0]), int(best[1])], float(scores[best])

    def _grid_3d(self, arrays: Sequence[np.ndarray], load_gamma: np.ndarray, freq_mask: np.ndarray):
        a0, a1, a2 = arrays
        total = a0.shape[0] * a1.shape[0] * a2.shape[0]
        sample = 1 if total <= 200000 else max(2, int(round(total ** (1 / 3) / 18)))
        i0s = np.arange(0, a0.shape[0], sample)
        i1s = np.arange(0, a1.shape[0], sample)
        i2s = np.arange(0, a2.shape[0], sample)

        b0, b1, b2 = a0[i0s], a1[i1s], a2[i2s]
        denom01 = 1.0 / (1.0 - b0[:, None, :, 1, 1] * b1[None, :, :, 0, 0] + 1e-50)
        s11_01 = b0[:, None, :, 0, 0] + b0[:, None, :, 0, 1] * b1[None, :, :, 0, 0] * b0[:, None, :, 1, 0] * denom01
        s21_01 = b0[:, None, :, 1, 0] * b1[None, :, :, 1, 0] * denom01
        s12_01 = b0[:, None, :, 0, 1] * b1[None, :, :, 0, 1] * denom01
        s22_01 = b1[None, :, :, 1, 1] + b1[None, :, :, 0, 1] * b0[:, None, :, 1, 1] * b1[None, :, :, 1, 0] * denom01

        denom = 1.0 / (1.0 - s22_01[:, :, None, :] * b2[None, None, :, :, 0, 0] + 1e-50)
        s21 = s21_01[:, :, None, :] * b2[None, None, :, :, 1, 0] * denom
        s22 = b2[None, None, :, :, 1, 1] + b2[None, None, :, :, 0, 1] * s22_01[:, :, None, :] * b2[None, None, :, :, 1, 0] * denom
        gt = (
            np.abs(s21) ** 2
            * np.maximum(0.0, 1.0 - np.abs(load_gamma) ** 2)
            / (np.abs(1.0 - s22 * load_gamma) ** 2 + 1e-50)
        )
        scores = np.mean(gt[..., freq_mask], axis=-1)
        coarse = np.unravel_index(int(np.argmax(scores)), scores.shape)
        best = (int(i0s[coarse[0]]), int(i1s[coarse[1]]), int(i2s[coarse[2]]))
        best_score = float(scores[coarse])

        radius = max(1, sample)
        ranges = [
            range(max(0, best[p] - radius), min(arrays[p].shape[0], best[p] + radius + 1))
            for p in range(3)
        ]
        for i in ranges[0]:
            for j in ranges[1]:
                s12 = _cascade_2port(a0[i], a1[j])
                for k in ranges[2]:
                    score = _vector_score(_transducer_gain(_cascade_2port(s12, a2[k]), load_gamma)[freq_mask])
                    if score > best_score:
                        best_score = score
                        best = (i, j, k)
        return list(best), best_score

    def _grid_nd_coarse(self, arrays: Sequence[np.ndarray], load_gamma: np.ndarray, freq_mask: np.ndarray):
        # Small fallback for 2-element or unusual requests.
        best = [0] * len(arrays)
        best_score = -1.0
        limits = [range(a.shape[0]) for a in arrays]
        for indices in np.array(np.meshgrid(*[np.arange(len(r)) for r in limits])).T.reshape(-1, len(arrays)):
            s = _cascade_network([arrays[pos][idx] for pos, idx in enumerate(indices)], self.nf)
            score = _vector_score(_transducer_gain(s, load_gamma)[freq_mask])
            if score > best_score:
                best_score = score
                best = [int(i) for i in indices]
        return best, best_score

    def optimize(self, antenna_s: np.ndarray, port_configs: Sequence[dict], max_rounds: int = 4):
        current = {}
        thru = np.zeros((self.nf, 2, 2), dtype=complex)
        thru[:, 0, 1] = 1.0
        thru[:, 1, 0] = 1.0
        for pc in port_configs:
            current[pc["port"]] = {"S": thru, "port_type": pc["port_type"], "ground_term": pc.get("ground_term", "short")}

        combos_cache = {}
        for pc in port_configs:
            n = int(pc.get("max_components", pc.get("elem_count", 3)))
            topo_allowed = pc.get("topologies")
            combos = []
            for topo in topo_allowed or _topologies(n):
                for types in _type_assignments(len(topo)):
                    combos.append((tuple(topo), tuple(types)))
            combos_cache[pc["port"]] = combos

        best_score = -1.0
        best_state = None
        load_ports = [pc for pc in port_configs if pc["port_type"] == "load"]
        ground_ports = [pc for pc in port_configs if pc["port_type"] == "ground"]
        for _ in range(max_rounds):
            for pc in load_ports:
                ap = pc["port"]
                gamma = {}
                for other in port_configs:
                    op = other["port"]
                    if op == ap:
                        continue
                    gamma[op] = _port_reflection(current[op]["S"], current[op]["port_type"], current[op].get("ground_term", "short"))
                load_gamma = _effective_1port(antenna_s, ap, gamma)
                best = None
                for topo, types in combos_cache[ap]:
                    trial = self._best_for_port(topo, types, load_gamma, pc.get("freq_mask"))
                    if best is None or trial["score"] > best["score"]:
                        best = trial
                current[ap] = {
                    **best,
                    "S": _cascade_network(best["s_list"], self.nf),
                    "port_type": "load",
                    "ground_term": pc.get("ground_term", "short"),
                }

            for pc in ground_ports:
                ap = pc["port"]
                best = None
                for topo, types in combos_cache[ap]:
                    # Ground ports are optimized by impact on load-port accepted power.
                    trial = self._best_for_port(topo, types, np.zeros(self.nf, dtype=complex), pc.get("freq_mask"))
                    s_trial = _cascade_network(trial["s_list"], self.nf)
                    score = self._combined_score(antenna_s, current, port_configs, override=(ap, s_trial, pc))
                    trial["score"] = score
                    if best is None or trial["score"] > best["score"]:
                        best = trial
                current[ap] = {
                    **best,
                    "S": _cascade_network(best["s_list"], self.nf),
                    "port_type": "ground",
                    "ground_term": pc.get("ground_term", "short"),
                }

            score = self._combined_score(antenna_s, current, port_configs)
            if score > best_score:
                best_score = score
                best_state = {k: dict(v) for k, v in current.items()}

        return self._build_result(antenna_s, best_state or current, port_configs, best_score)

    def _combined_score(self, antenna_s, current, port_configs, override=None):
        values = []
        for pc in port_configs:
            if pc["port_type"] != "load":
                continue
            gamma = {}
            for other in port_configs:
                op = other["port"]
                if op == pc["port"]:
                    continue
                if override is not None and override[0] == op:
                    s_match, port_type, ground_term = override[1], override[2]["port_type"], override[2].get("ground_term", "short")
                else:
                    s_match, port_type, ground_term = current[op]["S"], current[op]["port_type"], current[op].get("ground_term", "short")
                gamma[op] = _port_reflection(s_match, port_type, ground_term)
            load_gamma = _effective_1port(antenna_s, pc["port"], gamma)
            mask = pc.get("freq_mask")
            if mask is None:
                mask = np.ones(self.nf, dtype=bool)
            values.append(_vector_score(_transducer_gain(current[pc["port"]]["S"], load_gamma)[mask]))
        if not values:
            return 0.0
        return len(values) / sum(1.0 / max(v, 1e-10) for v in values)

    def _build_result(self, antenna_s, state, port_configs, score):
        per_port = []
        for pc in port_configs:
            ap = pc["port"]
            net = state[ap]
            gamma = {}
            for other in port_configs:
                op = other["port"]
                if op == ap:
                    continue
                gamma[op] = _port_reflection(state[op]["S"], state[op]["port_type"], state[op].get("ground_term", "short"))
            load_gamma = _effective_1port(antenna_s, ap, gamma)
            gt = _transducer_gain(net["S"], load_gamma)
            gin = _input_gamma(net["S"], load_gamma)
            accepted = np.maximum(0.0, 1.0 - np.abs(gin) ** 2)
            mask = pc.get("freq_mask")
            if mask is None:
                mask = np.ones(self.nf, dtype=bool)
            per_port.append({
                "port": ap,
                "port_type": net["port_type"],
                "topology": net.get("topology", []),
                "type_codes": net.get("type_codes", []),
                "types": net.get("types", []),
                "values": net.get("values", []),
                "part_numbers": net.get("part_numbers", []),
                "components": net.get("components", []),
                "s11_complex": gin,
                "s11_mag": float(np.mean(np.abs(gin[mask]))),
                "s11_db": float(np.mean(20.0 * np.log10(np.abs(gin[mask]) + 1e-12))),
                "accepted_efficiency": float(np.mean(accepted[mask])),
                "transducer_gain_estimate": float(np.mean(gt[mask])),
            })
        return {"combined_score": score, "per_port": per_port}


def _topologies(n: int) -> Iterable[Tuple[str, ...]]:
    if n <= 0:
        return [()]
    return [tuple(t) for t in __import__("itertools").product(("S", "P"), repeat=n)]


def _type_assignments(n: int) -> Iterable[Tuple[str, ...]]:
    return [tuple(t) for t in __import__("itertools").product(("L", "C"), repeat=n)]


def _representative_db_components(series_name: str, component_type: str, db_path: Path) -> List[GridComponent]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT c.part_number, c.nominal_value, c.nominal_unit, c.zip_path, c.is_primary
        FROM components c
        JOIN series s ON c.series_id = s.id
        WHERE s.name = ? AND c.component_type = ?
        ORDER BY c.nominal_value, c.is_primary DESC, c.part_number
        """,
        (series_name, component_type),
    ).fetchall()
    by_value = {}
    for row in rows:
        value = float(row["nominal_value"])
        path = Path(row["zip_path"])
        if value not in by_value and path.exists():
            by_value[value] = GridComponent(
                part_number=row["part_number"],
                component_type=component_type,
                nominal_value=value,
                nominal_unit=row["nominal_unit"],
                s2p_path=str(path),
            )
    return [by_value[v] for v in sorted(by_value)]


def load_gjm_lqp_libraries(freqs_hz: np.ndarray, db_path: Path = DEFAULT_DB_PATH):
    inductors = _representative_db_components("LQP03HQ_02", "inductor", db_path)
    capacitors = _representative_db_components("GJM03", "capacitor", db_path)

    def build(parts: List[GridComponent]):
        out = []
        for comp in parts:
            data = load_touchstone_file(comp.s2p_path)
            s = np.stack([data.get_s_matrix_interpolated(float(f)) for f in freqs_hz])
            y = _s_to_y(s)
            out.append((comp, s, s, _shunt_s({"y": y})))
        return out

    return build(inductors), build(capacitors)


def optimize_narrowband_grid(
    dut: TouchstoneData,
    port_specs: Sequence[dict],
    num_band_points: int = 1,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict:
    start = time.time()
    enabled = [p for p in port_specs if p.get("enabled", True)]
    all_bands = []
    for spec in enabled:
        all_bands.extend(spec.get("bands_mhz", [[2400, 2500]]))
    freqs_hz = _band_points_hz(all_bands, num_band_points)
    antenna_s = np.stack([dut.get_s_matrix_interpolated(float(f)) for f in freqs_hz])
    l_lib, c_lib = load_gjm_lqp_libraries(freqs_hz, db_path=db_path)

    port_configs = []
    for idx, spec in enumerate(enabled):
        port_type = spec.get("port_type")
        if port_type is None:
            port_type = "load" if idx == 0 else "ground"
        if spec.get("port_type") is None:
            max_components = 3 if port_type == "load" else 1
        else:
            max_components = int(spec.get("max_components", 3 if port_type == "load" else 1))
        port_configs.append({
            "port": int(spec.get("port_index", idx)),
            "port_type": port_type,
            "max_components": max_components,
            "ground_term": spec.get("ground_term", "short"),
            "bands_mhz": spec.get("bands_mhz", [[2400, 2500]]),
        })

    for pc in port_configs:
        mask = np.zeros(len(freqs_hz), dtype=bool)
        for band in pc.get("bands_mhz", []):
            start_hz = float(band[0]) * 1e6
            stop_hz = float(band[1]) * 1e6
            mask |= (freqs_hz >= min(start_hz, stop_hz) - 1e-6) & (freqs_hz <= max(start_hz, stop_hz) + 1e-6)
        if not np.any(mask):
            mask[:] = True
        pc["freq_mask"] = mask

    optimizer = NarrowbandGridS2P(l_lib, c_lib, freqs_hz)
    result = optimizer.optimize(antenna_s, port_configs)
    result.update({
        "freqs_hz": freqs_hz,
        "time_sec": time.time() - start,
        "library": {"inductors": len(l_lib), "capacitors": len(c_lib)},
        "efficiency_basis": "accepted_power_and_transducer_gain_estimate_not_radiation_efficiency",
    })
    return result
