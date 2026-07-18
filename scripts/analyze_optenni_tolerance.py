"""Infer Optenni generic L/C tolerance samples from a native plot export."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "rfmatch-core" / "src"))

from rfmatch_core import load_optenni_tolerance_export, load_touchstone


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _network_values(network_path: Path) -> tuple[float, float]:
    network = load_touchstone(network_path)
    if network.s_parameters.shape[1:] != (2, 2):
        raise ValueError("matching network must be an S2P")
    s = network.s_parameters
    s11, s12 = s[:, 0, 0], s[:, 0, 1]
    s21, s22 = s[:, 1, 0], s[:, 1, 1]
    z0 = network.z0
    # Series-Z followed by shunt-Y has ABCD B=Z and C=Y.
    abcd_b = z0 * ((1 + s11) * (1 + s22) - s12 * s21) / (2 * s21)
    abcd_c = ((1 - s11) * (1 - s22) - s12 * s21) / (2 * z0 * s21)
    omega = 2 * np.pi * network.frequencies_hz
    return float(np.median(abcd_b.imag / omega)), float(np.median(abcd_c.imag / omega))


def _responses(
    frequencies_hz: np.ndarray,
    dut_s11: np.ndarray,
    inductance_h: float,
    capacitance_f: float,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    z0 = 50.0
    omega = 2 * np.pi * frequencies_hz
    dut_admittance = 1.0 / (z0 * (1 + dut_s11) / (1 - dut_s11))
    varied_l = inductance_h * scales[0]
    varied_c = capacitance_f * scales[1]
    # Optenni project settings: inductor Q=30 at 1 GHz and capacitor ESR=0.4 ohm.
    inductor_resistance = 2 * np.pi * 1e9 * varied_l / 30.0
    inductor_impedance = inductor_resistance + 1j * omega * varied_l
    capacitor_impedance = 0.4 + 1.0 / (1j * omega * varied_c)
    input_impedance = inductor_impedance + 1.0 / (
        dut_admittance + 1.0 / capacitor_impedance
    )
    gamma = (input_impedance - z0) / (input_impedance + z0)
    input_voltage = np.sqrt(z0) * (1 + gamma)
    input_current = (1 - gamma) / np.sqrt(z0)
    dut_voltage = input_voltage - input_current * inductor_impedance
    efficiency = np.abs(dut_voltage) ** 2 * np.real(dut_admittance)
    return (
        20 * np.log10(np.maximum(np.abs(gamma), 1e-15)),
        10 * np.log10(np.maximum(efficiency, 1e-15)),
    )


def _fit_sample(residual, iterations: int = 8) -> tuple[np.ndarray, float]:
    scales = np.ones(2)
    for _ in range(iterations):
        value = residual(scales)
        step = 1e-5
        jacobian = np.column_stack([
            (residual(scales + np.eye(2)[column] * step) - value) / step
            for column in range(2)
        ])
        delta = np.linalg.lstsq(jacobian, -value, rcond=None)[0]
        scales = np.clip(scales + delta, 0.95, 1.05)
        if float(np.max(np.abs(delta))) < 1e-11:
            break
    return scales, float(np.sqrt(np.mean(residual(scales) ** 2)))


def _uniform_ks(values: np.ndarray, lower: float, upper: float) -> tuple[float, float]:
    normalized = np.sort(np.clip((values - lower) / (upper - lower), 0.0, 1.0))
    count = len(normalized)
    empirical_upper = np.arange(1, count + 1) / count
    empirical_lower = np.arange(count) / count
    statistic = float(max(np.max(empirical_upper - normalized), np.max(normalized - empirical_lower)))
    adjusted = (np.sqrt(count) + 0.12 + 0.11 / np.sqrt(count)) * statistic
    terms = [(-1) ** (index - 1) * np.exp(-2 * index * index * adjusted * adjusted) for index in range(1, 101)]
    return statistic, float(np.clip(2 * sum(terms), 0.0, 1.0))


def analyze(dut_path: Path, network_path: Path, export_path: Path) -> dict:
    dut = load_touchstone(dut_path)
    export = load_optenni_tolerance_export(export_path)
    if dut.s_parameters.shape[1:] != (1, 1):
        raise ValueError("DUT must be an S1P")
    if not np.allclose(dut.frequencies_hz, export.frequencies_hz, rtol=1e-12, atol=1.0):
        raise ValueError("DUT and tolerance export frequency grids differ")
    inductance_h, capacitance_f = _network_values(network_path)
    observed_efficiency_db = 10 * np.log10(export.total_efficiency_variants)
    fit_indices = np.arange(0, len(export.frequencies_hz), 5)
    fitted, residual_rms = [], []
    for sample in range(export.samples):
        def residual(scales):
            s11_db, efficiency_db = _responses(
                export.frequencies_hz,
                dut.s_parameters[:, 0, 0],
                inductance_h,
                capacitance_f,
                scales,
            )
            return np.concatenate((
                s11_db[fit_indices] - export.s11_variants_db[fit_indices, sample],
                efficiency_db[fit_indices] - observed_efficiency_db[fit_indices, sample],
            ))
        scales, rms = _fit_sample(residual)
        fitted.append(scales)
        residual_rms.append(rms)
    fitted = np.asarray(fitted)
    tolerance = 0.02
    components = {}
    for column, name in enumerate(("inductor", "capacitor")):
        values = fitted[:, column]
        statistic, p_value = _uniform_ks(values, 1 - tolerance, 1 + tolerance)
        components[name] = {
            "minimum_scale": float(np.min(values)),
            "maximum_scale": float(np.max(values)),
            "mean_scale": float(np.mean(values)),
            "sample_standard_deviation": float(np.std(values, ddof=1)),
            "uniform_expected_standard_deviation": tolerance / np.sqrt(3),
            "uniform_ks_statistic": statistic,
            "uniform_ks_asymptotic_p_value": p_value,
            "percentiles": {
                str(p): float(value)
                for p, value in zip(
                    (0, 1, 5, 25, 50, 75, 95, 99, 100),
                    np.percentile(values, (0, 1, 5, 25, 50, 75, 95, 99, 100)),
                )
            },
        }
    return {
        "samples": export.samples,
        "inferred_nominal_inductance_nh": inductance_h * 1e9,
        "inferred_nominal_capacitance_pf": capacitance_f * 1e12,
        "configured_tolerance_fraction": tolerance,
        "inferred_distribution": "independent uniform within configured bounds",
        "component_scale_correlation": float(np.corrcoef(fitted.T)[0, 1]),
        "fit_rms_db_median": float(np.median(residual_rms)),
        "fit_rms_db_maximum": float(np.max(residual_rms)),
        "components": components,
        "inputs": {
            "dut": {"path": str(dut_path), "sha256": _sha256(dut_path)},
            "matching_network": {"path": str(network_path), "sha256": _sha256(network_path)},
            "tolerance_export": {"path": str(export_path), "sha256": _sha256(export_path)},
        },
    }


def main() -> int:
    baseline = ROOT / "benchmarks" / "optenni_exports"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dut", type=Path, default=baseline / "optimization_settings_original.s1p")
    parser.add_argument("--network", type=Path, default=baseline / "optimization_settings_pcsl_circuit.s2p")
    parser.add_argument("--export", type=Path, default=baseline / "optimization_settings_pcsl_tolerance_100.txt")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = analyze(args.dut, args.network, args.export)
    except (OSError, ValueError, np.linalg.LinAlgError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
