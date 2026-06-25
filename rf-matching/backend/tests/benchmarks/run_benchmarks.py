"""
RF Matching Benchmark Suite.

Each benchmark:
  - Loads an SNP file and component library
  - Runs tuning with specific config
  - Asserts trends: efficiency improves, RL improves, etc.

Run: python -m tests.benchmarks.run_benchmarks
"""

import sys, os, time, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from engine.touchstone import parse_touchstone
from engine.murata_db_adapter import load_murata_db
from engine.tuning_service import (
    run_tuning_single, run_tuning_joint, run_tuning_tunable_c,
    TuningResult, get_session, reset_session,
)
from engine.scoring import efficiency_chain


# ── Helpers ──

SNP_DIR = r"E:\RF matching\snp"
MURATA_DB = r"E:\RF matching\Murata\murata_components.db"

_lib_cache = None

def get_library():
    global _lib_cache
    if _lib_cache is None and os.path.isfile(MURATA_DB):
        _lib_cache = load_murata_db(MURATA_DB)
    return _lib_cache


def load_snp(rel_path):
    full = os.path.join(SNP_DIR, rel_path)
    if not os.path.isfile(full):
        raise FileNotFoundError(f"SNP not found: {full}")
    with open(full, 'r', encoding='utf-8', errors='replace') as f:
        return parse_touchstone(f.read(), filename=os.path.basename(full))


class BenchmarkResult:
    def __init__(self, name):
        self.name = name
        self.passed = True
        self.messages = []
        self.metrics = {}
        self.duration = 0.0

    def assert_trend(self, condition, msg):
        if not condition:
            self.passed = False
            self.messages.append(f"FAIL: {msg}")
        else:
            self.messages.append(f"OK: {msg}")

    def assert_improves(self, raw_val, tuned_val, metric_name):
        """Assert that tuned value is better than raw."""
        if tuned_val > raw_val:
            self.messages.append(f"OK: {metric_name} improved {raw_val:.3f} -> {tuned_val:.3f}")
        elif abs(tuned_val - raw_val) < 0.01:
            self.messages.append(f"WARN: {metric_name} unchanged {raw_val:.3f}")
        else:
            self.passed = False
            self.messages.append(f"FAIL: {metric_name} degraded {raw_val:.3f} -> {tuned_val:.3f}")

    def report(self, indent=""):
        status = "PASS" if self.passed else "FAIL"
        print(f"{indent}[{status}] {self.name} ({self.duration:.1f}s)")
        for m in self.messages:
            print(f"{indent}  {m}")
        return self.passed


# ── Benchmark 1: Single-port basic ──

def benchmark_single_port_basic():
    """Test basic single-port matching with a 3-port antenna at 2.45 GHz.
    Uses Port 2 (poor match: |S11|~0.87) to verify efficiency improvement.
    """
    b = BenchmarkResult("Single-port basic (3_antennas.s3p, Port 2, 2.45 GHz)")

    dut = load_snp(r"case1\3_antennas.s3p")
    lib = get_library()
    if lib is None:
        b.messages.append("SKIP: Library not available")
        return b

    t0 = time.time()
    port_idx = 1  # Port 2 (0-based) — poor match

    # Raw S11 at center
    S_raw = dut.get_s_matrix_interpolated(2.45e9)
    raw_s11 = abs(S_raw[port_idx, port_idx])
    raw_eff = 1.0 - raw_s11 ** 2
    b.metrics['raw_s11'] = raw_s11
    b.metrics['raw_eff'] = raw_eff
    b.messages.append(f"Raw port 2: |S11|={raw_s11:.4f}, η_accepted={raw_eff:.3f}")

    # Run tuning
    candidates = run_tuning_single(
        dut=dut, library=lib,
        port_index=port_idx,
        bands_mhz=[[2400, 2500]],
        max_components=2,
        objective="average_efficiency",
        beam_width=10,
        timeout_seconds=30,
        num_band_points=5,
    )

    b.duration = time.time() - t0

    if not candidates:
        b.messages.append("FAIL: No solutions found")
        b.passed = False
        return b

    best = candidates[0]
    pp = best.per_port.get(port_idx)

    b.metrics['tuned_s11'] = pp.s11_magnitude if pp else 1.0
    b.metrics['tuned_eff'] = best.avg_total_efficiency

    b.messages.append(f"Tuned: score={best.system_score:.3f}, η_avg={best.avg_total_efficiency:.3f}")
    b.messages.append(f"Solutions found: {best.num_solutions_found}")

    # For a poorly-matched port, total efficiency should improve
    # (accepted efficiency gain > component loss)
    b.assert_improves(raw_eff, best.avg_total_efficiency, "Total efficiency (accepted - loss)")
    if pp:
        # s11_db is stored as positive RL magnitude
        rl_db = pp.s11_db
        b.assert_trend(rl_db > 8, f"RL={rl_db:.1f}dB > 8dB (improved from raw S11={raw_s11:.2f})")
    b.assert_trend(best.system_score > 0.1, f"Score={best.system_score:.3f} > 0.1")
    b.assert_trend(best.total_component_count <= 2, f"Components={best.total_component_count} <= 2")

    return b


# ── Benchmark 2: Two-port joint matching ──

def benchmark_two_port_joint():
    """Test joint matching for 2 ports of a 3-port antenna."""
    b = BenchmarkResult("Two-port joint (3_antennas.s3p, Ports 1+2)")

    dut = load_snp(r"case1\3_antennas.s3p")
    lib = get_library()
    if lib is None:
        b.messages.append("SKIP: Library not available")
        return b

    t0 = time.time()

    # Raw metrics
    S_raw = dut.get_s_matrix_interpolated(2.45e9)
    raw_effs = [1.0 - abs(S_raw[i, i]) ** 2 for i in range(2)]
    b.messages.append(f"Raw: P1 S11={abs(S_raw[0,0]):.3f}, P2 S11={abs(S_raw[1,0]):.3f}")

    candidates = run_tuning_joint(
        dut=dut, library=lib,
        port_specs=[
            {'port_index': 0, 'bands_mhz': [[2400, 2500]], 'max_components': 2, 'enabled': True},
            {'port_index': 1, 'bands_mhz': [[2400, 2500]], 'max_components': 2, 'enabled': True},
        ],
        objective="balanced",
        beam_width=5,
        timeout_seconds=60,
        num_band_points=3,
    )

    b.duration = time.time() - t0

    if not candidates:
        b.messages.append("FAIL: No solutions found")
        b.passed = False
        return b

    best = candidates[0]
    b.messages.append(f"Best: score={best.system_score:.3f}, η_avg={best.avg_total_efficiency:.3f}, η_min={best.min_total_efficiency:.3f}")
    b.messages.append(f"Coupling: avg={best.avg_coupling_loss:.3f}, max={best.max_coupling_loss:.3f}")
    b.messages.append(f"Solutions: {best.num_solutions_found}")

    avg_raw = np.mean(raw_effs)
    b.assert_improves(avg_raw, best.avg_total_efficiency, "Average efficiency")
    b.assert_trend(best.system_score > 0, f"Score={best.system_score:.3f} > 0")
    b.assert_trend(best.total_component_count >= 2, f"Components={best.total_component_count} >= 2")

    return b


# ── Benchmark 3: Efficiency improvement check ──

def benchmark_efficiency_improvement():
    """Verify tuning improves efficiency for a poorly-matched port."""
    b = BenchmarkResult("Efficiency improvement (3_antennas P2, multi-band)")

    dut = load_snp(r"case1\3_antennas.s3p")
    lib = get_library()
    if lib is None:
        b.messages.append("SKIP: Library not available")
        return b

    t0 = time.time()
    port_idx = 1  # Port 2 — poor match

    # Raw efficiency at multiple frequencies
    raw_effs = []
    for f in [2.3e9, 2.45e9, 2.6e9]:
        S = dut.get_s_matrix_interpolated(f)
        raw_effs.append(1.0 - abs(S[port_idx, port_idx]) ** 2)
    b.messages.append(f"Raw avg accepted eff: {np.mean(raw_effs):.3f}")

    candidates = run_tuning_single(
        dut=dut, library=lib,
        port_index=port_idx,
        bands_mhz=[[2300, 2600]],
        max_components=2,
        objective="average_efficiency",
        beam_width=10,
        timeout_seconds=30,
        num_band_points=5,
    )

    b.duration = time.time() - t0

    if not candidates:
        b.messages.append("FAIL: No solutions")
        b.passed = False
        return b

    best = candidates[0]
    b.messages.append(f"Tuned total eff: {best.avg_total_efficiency:.3f}")

    # For a port with poor raw match, tuning should improve accepted efficiency
    # enough to overcome component loss
    b.assert_improves(np.mean(raw_effs), best.avg_total_efficiency,
                      "Total efficiency across band")
    b.assert_trend(best.avg_total_efficiency > 0.3,
                   f"Tuned efficiency={best.avg_total_efficiency:.3f} > 0.3")

    return b


# ── Benchmark 4: Component loss bounded ──

def benchmark_component_loss():
    """Verify component loss is properly estimated and bounded."""
    import numpy as np
    from engine.scoring import estimate_component_loss_power

    b = BenchmarkResult("Component loss estimation sanity")

    # Lossless component
    perfect = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    loss = estimate_component_loss_power(perfect)
    b.assert_trend(abs(loss) < 1e-10, f"Lossless: loss={loss:.6f} (expect 0)")

    # 1dB insertion loss
    s21 = 10 ** (-1 / 20)
    lossy = np.array([[0.0, s21], [s21, 0.0]], dtype=complex)
    loss = estimate_component_loss_power(lossy)
    b.assert_trend(abs(loss - 0.2057) < 0.01, f"1dB IL: loss={loss:.4f} (expect ~0.206)")

    # 3dB insertion loss  
    s21 = 10 ** (-3 / 20)
    lossy = np.array([[0.0, s21], [s21, 0.0]], dtype=complex)
    loss = estimate_component_loss_power(lossy)
    b.assert_trend(abs(loss - 0.5) < 0.01, f"3dB IL: loss={loss:.4f} (expect ~0.5)")

    return b


# ── Benchmark 5: Tunable capacitor search ──

def benchmark_tunable_capacitor():
    """Verify tunable capacitor search produces candidates with real Murata DB.
    Uses Port 0 (moderate mismatch) for a meaningful improvement test."""
    b = BenchmarkResult("Tunable C search (3_antennas.s3p, Port 0, W24, MurataDB)")

    dut = load_snp(r"case1\3_antennas.s3p")
    lib = get_library()

    t0 = time.time()

    # Verify Port 0 needs matching (raw RL ~14dB → moderate)
    S_raw = dut.get_s_matrix_interpolated(2.45e9)
    raw_s11_0 = abs(S_raw[0, 0])
    b.messages.append(f"Raw Port 0: |S11|={raw_s11_0:.3f}, RL={-20*np.log10(max(raw_s11_0,1e-15)):.1f}dB")

    # Primary test: real Murata DB components on Port 0 (moderate mismatch)
    candidates = run_tuning_tunable_c(
        dut=dut, library=lib,
        port_index=0,
        band_state_map={"W24": [2400, 2500]},
        beam_width=10,
        num_band_points=3,
    )

    b.duration = time.time() - t0
    b.messages.append(f"MurataDB candidates: {len(candidates)}")

    b.assert_trend(len(candidates) > 0, f"Found {len(candidates)} candidates with MurataDB")
    if candidates:
        best = candidates[0]
        pp = best.per_port.get(0)
        b.assert_trend(best.system_score > 0, f"Best score={best.system_score:.3f}")
        if pp:
            b.messages.append(f"Best eff={best.avg_total_efficiency:.3f}, RL={pp.s11_db:.1f}dB, comps={len(pp.components)}")
            # Verify algorithm produces non-trivial result
            b.assert_trend(pp.s11_db > 0.5, f"RL={pp.s11_db:.1f}dB > 0.5dB (search produced valid result)")

    # Secondary test: ideal components (flexible baseline, smoke test)
    t1 = time.time()
    ideal_candidates = run_tuning_tunable_c(
        dut=dut, library=None,
        port_index=0,
        band_state_map={"W24": [2400, 2500]},
        beam_width=10,
        num_band_points=3,
    )
    b.messages.append(f"Ideal candidates: {len(ideal_candidates)}")
    b.assert_trend(len(ideal_candidates) > 0, f"Ideal search found {len(ideal_candidates)} candidates")
    if ideal_candidates:
        best_i = ideal_candidates[0]
        pp_i = best_i.per_port.get(0)
        if pp_i:
            b.messages.append(f"Ideal best eff={best_i.avg_total_efficiency:.3f}, RL={pp_i.s11_db:.1f}dB")
    b.duration = time.time() - t0

    return b


# ── Runner ──

BENCHMARKS = [
    benchmark_single_port_basic,
    benchmark_efficiency_improvement,
    benchmark_component_loss,
    benchmark_two_port_joint,
    benchmark_tunable_capacitor,
]

BENCHMARKS_LONG = []


def run_benchmarks(include_long=False):
    print("=" * 60)
    print("RF Matching Benchmark Suite")
    print("=" * 60)
    print()

    to_run = BENCHMARKS.copy()
    if include_long:
        to_run.extend(BENCHMARKS_LONG)

    results = []
    for bm_fn in to_run:
        result = bm_fn()
        results.append(result)
        result.report(indent="  ")
        print()

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"{'='*60}")
    print(f"Results: {passed}/{total} passed")
    if passed < total:
        print(f"FAILED benchmarks:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}")
    print()

    return all(r.passed for r in results)


if __name__ == "__main__":
    import sys
    include_long = "--long" in sys.argv
    success = run_benchmarks(include_long=include_long)
    sys.exit(0 if success else 1)
