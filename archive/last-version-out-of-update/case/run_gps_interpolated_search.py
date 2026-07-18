import importlib.util
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np


CASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = Path(r"E:\RF matching\rf-matching\backend")
DB_PATH = Path(r"E:\RF matching\Murata\optenni_components.db")
ANTENNA_PATH = CASE_DIR / "GPS GND.s2p"
ENGINE_PATH = CASE_DIR / "engine grid s2p - nport.py"
TARGET_FREQS_HZ = np.array([1176.45e6, 1575.42e6], dtype=float)
BANDS_MHZ = [(1176.45, 1176.45), (1575.42, 1575.42)]


sys.path.insert(0, str(BACKEND_DIR))
from engine.touchstone import load_touchstone_file  # noqa: E402


def load_grid_engine():
    spec = importlib.util.spec_from_file_location("grid_s2p_nport", ENGINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def s_to_y(s_matrix, z0=50.0):
    """Convert interpolated 2-port S matrices to Y matrices."""
    ident = np.eye(s_matrix.shape[-1], dtype=complex)
    y0 = 1.0 / z0
    out = np.empty_like(s_matrix)
    for idx, s in enumerate(s_matrix):
        out[idx] = y0 * (ident - s) @ np.linalg.pinv(ident + s)
    return out


def interpolate_file(path, freqs_hz):
    data = load_touchstone_file(str(path))
    return np.stack([data.get_s_matrix_interpolated(float(freq)) for freq in freqs_hz])


def representative_rows(series_name, component_type):
    """Pick one S2P per nominal value, preferring primary parts when present."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT c.part_number, c.nominal_value, c.zip_path, c.is_primary
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
        if value not in by_value and Path(row["zip_path"]).exists():
            by_value[value] = row
    return [by_value[value] for value in sorted(by_value)]


def load_component_library(series_name, component_type):
    lib = []
    for row in representative_rows(series_name, component_type):
        s_matrix = interpolate_file(row["zip_path"], TARGET_FREQS_HZ)
        lib.append(
            (
                float(row["nominal_value"]),
                row["part_number"],
                {"s": s_matrix, "y": s_to_y(s_matrix)},
            )
        )
    return lib


def main():
    engine = load_grid_engine()

    t0 = time.perf_counter()
    antenna_s = interpolate_file(ANTENNA_PATH, TARGET_FREQS_HZ)
    l_library = load_component_library("LQP03HQ_02", "inductor")
    c_library = load_component_library("GJM03", "capacitor")
    load_sec = time.perf_counter() - t0

    print(f"Interpolated frequencies MHz: {[f / 1e6 for f in TARGET_FREQS_HZ]}")
    print(f"Library: {len(l_library)} LQP03HQ_02 values, {len(c_library)} GJM03 values")
    print(f"SNP interpolation/load time: {load_sec:.3f} s")

    port_configs = [
        {
            "port": 0,
            "port_type": "load",
            "elem_min": 3,
            "elem_max": 3,
            "bands": BANDS_MHZ,
            "reverse": "0",
        },
        {
            "port": 1,
            "port_type": "ground",
            "elem_min": 1,
            "elem_max": 1,
            "ground_term": "short",
            "bands": BANDS_MHZ,
            "reverse": "1",
        },
    ]

    optimizer = engine.GridS2PforNport(l_library, c_library, TARGET_FREQS_HZ)
    t1 = time.perf_counter()
    result = optimizer.optimize(
        antenna_s,
        port_configs,
        TARGET_FREQS_HZ,
        BANDS_MHZ,
        elem_count=3,
        max_rounds=5,
        convergence_threshold=1e-5,
    )
    opt_sec = time.perf_counter() - t1

    best = result["best"]
    print(f"Optimization wall time: {opt_sec:.3f} s")
    print(f"Engine reported time: {result.get('time_sec', 0):.3f} s")
    print(f"Evaluated topology/type combos: {result.get('n_evaluated')}")
    print(f"Rounds: {result.get('n_rounds')}")
    print(f"Combined GT: {best['combined_score']:.8f}")
    print(f"Combined GT dB: {best['combined_gt_db']:.3f} dB")
    print(f"Combined efficiency: {best['combined_eff_pct']:.3f} %")

    for port in best["per_port"]:
        print(
            "PORT {port} {port_type}: topo={topology} types={comp_types} "
            "values={values} parts={part_numbers} mean_gt={mean_gt:.8f} "
            "gt_db={gt_db:.3f} s11_db={s11_db:.3f}".format(**port)
        )


if __name__ == "__main__":
    main()
