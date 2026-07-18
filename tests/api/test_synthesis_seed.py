"""Product integration regression for loss-aware full-band synthesis."""

import os
from pathlib import Path
import sys
import tempfile
import time
import zipfile

import numpy as np
import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "apps", "api"))

from engine.component_lib import ComponentInfo, ComponentLibrary
from engine.murata_db_adapter import load_murata_db
from engine.touchstone import load_touchstone_file, parse_touchstone
from engine.topology import get_standard_topologies
from engine.tuning_service import (
    _core_component_catalog,
    _loss_aware_single_seed,
    run_tuning_joint,
    run_tuning_single,
    normalize_allowed_topology_codes,
)
from rfmatch_core import load_component_model, renormalize_s_parameters
from reporting import _measured_search_status


def test_product_seed_recalls_optenni_pcsl_topology():
    root = Path(__file__).resolve().parents[2]
    path = root / "benchmarks" / "optenni_exports" / "optimization_settings_original.s1p"
    dut = load_touchstone_file(path)
    topologies = [
        topology for topology in get_standard_topologies()
        if topology.num_components == 2
    ]

    seed = _loss_aware_single_seed(
        dut=dut,
        port_index=0,
        bands_mhz=[[1700.0, 2500.0]],
        frequencies_hz=np.linspace(1.7e9, 2.5e9, 10).tolist(),
        topologies=topologies,
        objective="balanced",
    )

    assert seed["topology_signature"] == [["shunt", "C"], ["series", "L"]]
    assert seed["loss_model"] == {
        "inductor_q": 30.0,
        "inductor_q_reference_hz": 1e9,
        "inductor_esr": 0.0,
        "capacitor_esr": 0.4,
    }
    assert seed["maximum_power_balance_error"] < 1e-12
    assert seed["evaluations"] > 0


def test_product_seed_uses_radiation_tutorial_generic_loss_profile():
    root = Path(__file__).resolve().parents[2]
    dut = load_touchstone_file(
        root / "benchmarks" / "optenni_exports" / "optimization_settings_original.s1p"
    )
    seed = _loss_aware_single_seed(
        dut=dut,
        port_index=0,
        bands_mhz=[[1700.0, 2500.0]],
        frequencies_hz=np.linspace(1.7e9, 2.5e9, 5).tolist(),
        topologies=[
            topology for topology in get_standard_topologies()
            if topology.num_components == 2
        ],
        objective="balanced",
        generic_synthesis_loss={
            "inductor_q": 50.0,
            "inductor_q_reference_hz": 1e9,
            "inductor_esr_ohm": 0.0,
            "capacitor_esr_ohm": 0.3,
        },
    )
    assert seed["loss_model"] == {
        "inductor_q": 50.0,
        "inductor_q_reference_hz": 1e9,
        "inductor_esr": 0.0,
        "capacitor_esr": 0.3,
    }
    assert seed["requested_loss_model"]["scope"] == "continuous_topology_prior_only"


def _write_series_component(
    path: Path,
    kind: str,
    value_si: float,
    reference_resistance: float = 50.0,
) -> None:
    rows = [f"# HZ S RI R {reference_resistance:g}"]
    for frequency in (1.0e9, 2.0e9, 3.0e9):
        omega = 2.0 * np.pi * frequency
        impedance = 1j * omega * value_si if kind == "L" else 1.0 / (1j * omega * value_si)
        twice_z0 = 2.0 * reference_resistance
        reflection = impedance / (twice_z0 + impedance)
        transmission = twice_z0 / (twice_z0 + impedance)
        values = (reflection, transmission, transmission, reflection)
        rows.append(
            " ".join([f"{frequency:.0f}", *(
                token
                for value in values
                for token in (f"{value.real:.16g}", f"{value.imag:.16g}")
            )])
        )
    path.write_text("\n".join(rows) + "\n", encoding="ascii")


def _serialize_dut(matrices: np.ndarray, fmt: str, z0: float = 75.0) -> str:
    rows = [f"# GHZ S {fmt} R {z0:g}"]
    for frequency, matrix in zip((1.0, 2.0, 3.0), matrices):
        fields = [f"{frequency:g}"]
        for source in range(matrix.shape[1]):
            for destination in range(matrix.shape[0]):
                value = matrix[destination, source]
                if fmt == "RI":
                    pair = value.real, value.imag
                elif fmt == "MA":
                    pair = abs(value), np.rad2deg(np.angle(value))
                else:
                    pair = 20.0 * np.log10(abs(value)), np.rad2deg(np.angle(value))
                fields.extend(f"{item:.15g}" for item in pair)
        rows.append(" ".join(fields))
    return "\n".join(rows)


def test_fixed_single_workflow_uses_full_band_measured_physical_search():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        library = ComponentLibrary()
        for kind, values in (("L", (2.2, 5.6)), ("C", (0.5, 1.2))):
            for value in values:
                path = root / f"{kind}{value}.s2p"
                value_si = value * (1e-9 if kind == "L" else 1e-12)
                _write_series_component(path, kind, value_si)
                library.add_component(ComponentInfo(
                    part_number=path.stem,
                    s2p_filename=str(path),
                    zip_path="__DIR__",
                    component_type="inductor" if kind == "L" else "capacitor",
                    nominal_value=value,
                    nominal_unit="nH" if kind == "L" else "pF",
                ))
        dut = parse_touchstone(
            "# GHZ S RI R 50\n1.0 0.45 0.20\n2.0 0.45 0.20\n3.0 0.45 0.20\n",
            filename="synthetic.s1p",
        )

        candidates = run_tuning_single(
            dut=dut,
            library=library,
            port_index=0,
            bands_mhz=[[1000.0, 3000.0]],
            max_components=1,
            objective="balanced",
            beam_width=3,
            num_band_points=3,
            generic_synthesis_loss={
                "inductor_q": 50.0,
                "inductor_q_reference_hz": 1e9,
                "capacitor_esr_ohm": 0.3,
            },
        )

        assert candidates
        best = candidates[0]
        assert best.efficiency_basis == "rfmatch_core_physical_measured_s2p"
        assert best.search_diagnostics["measured_physical_search"] is True
        assert best.search_diagnostics["physical_evaluations"] > 0
        assert best.search_diagnostics["component_models_loaded"] > 0
        assert best.search_diagnostics["generic_synthesis_loss"] == {
            "inductor_q": 50.0,
            "inductor_q_reference_hz": 1e9,
            "inductor_esr_ohm": 0.0,
            "capacitor_esr_ohm": 0.3,
            "scope": "continuous_topology_prior_only",
        }
        assert best.maximum_power_balance_error < 1e-10
        assert len(best.per_port[0].band_total_eff) == 3


def test_product_single_topology_filter_is_enforced_by_measured_core():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        library = ComponentLibrary()
        for kind, value in (("L", 2.2), ("C", 1.2)):
            path = root / f"{kind}{value}.s2p"
            _write_series_component(
                path, kind, value * (1e-9 if kind == "L" else 1e-12)
            )
            library.add_component(ComponentInfo(
                part_number=path.stem, s2p_filename=str(path), zip_path="__DIR__",
                component_type="inductor" if kind == "L" else "capacitor",
                nominal_value=value, nominal_unit="nH" if kind == "L" else "pF",
            ))
        dut = parse_touchstone(
            "# GHZ S RI R 50\n1.0 0.45 0.20\n2.0 0.45 0.20\n3.0 0.45 0.20\n",
            filename="synthetic.s1p",
        )
        candidates = run_tuning_single(
            dut=dut, library=library, port_index=0,
            bands_mhz=[[1000.0, 3000.0]], max_components=2,
            objective="balanced", beam_width=4, timeout_seconds=5,
            num_band_points=3,
            topology_filter=["1-Element (Series-L)"],
        )

    assert candidates
    for candidate in candidates.values():
        components = candidate.per_port[0].components
        assert len(components) == 1
        assert components[0]["connection_type"] == "series"
        assert components[0]["type"] == "inductor"
        assert candidate.search_diagnostics["joint_refine_neighbors"] == 12
        assert candidate.search_diagnostics["joint_refine_port_blocks"] is True
        assert (
            candidate.search_diagnostics["joint_refine_port_block_max_components"]
            == 2
        )


def test_zero_component_single_baseline_uses_core_and_exact_full_band_power():
    dut = parse_touchstone(
        "# GHZ S RI R 50\n1.0 0.30 0.10\n2.0 0.20 -0.05\n3.0 0.10 0.00\n",
        filename="bare-dut.s1p",
    )
    candidates = run_tuning_single(
        dut=dut, library=ComponentLibrary(), port_index=0,
        bands_mhz=[[1000.0, 3000.0]], band_weights=[2.0], port_weight=3.0,
        max_components=0,
        objective="balanced", beam_width=3, timeout_seconds=1,
        num_band_points=3,
    )

    assert list(candidates) == [0]
    result = candidates[0]
    expected = np.array([1 - abs(0.30 + 0.10j) ** 2, 1 - abs(0.20 - 0.05j) ** 2, 1 - 0.10 ** 2])
    assert result.total_component_count == 0
    assert result.per_port[0].components == []
    assert np.allclose(result.per_port[0].band_total_eff, expected, atol=1e-12)
    assert result.search_diagnostics["numeric_core"] == "rfmatch_core"
    assert result.search_diagnostics["physical_core_evaluation"] is True
    assert result.search_diagnostics["measured_physical_search"] is False
    assert result.search_diagnostics["bare_dut_core_baseline"] is True
    assert result.efficiency_basis == "rfmatch_core_physical_bare_dut"
    assert result.search_diagnostics["topology_code"] == "0"
    assert result.search_diagnostics["component_models_loaded"] == 0
    assert result.search_diagnostics["priority_weights"] == {
        "port_weight": 3.0,
        "band_weights": [2.0],
        "effective_band_weights": [6.0],
        "semantics": "band_margin_multiplier",
    }
    assert result.maximum_power_balance_error < 1e-12
    assert _measured_search_status(result.to_dict()) == "full-band bare-DUT physical baseline (rfmatch_core)"


def test_single_type_measured_catalog_runs_without_legacy_fallback():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "L2.2.s2p"
        _write_series_component(path, "L", 2.2e-9)
        library = ComponentLibrary()
        library.add_component(ComponentInfo(
            part_number="L2.2", s2p_filename=str(path), zip_path="__DIR__",
            component_type="inductor", nominal_value=2.2, nominal_unit="nH",
        ))
        dut = parse_touchstone(
            "# GHZ S RI R 50\n1.0 0.45 0.20\n2.0 0.40 0.15\n3.0 0.35 0.10\n",
            filename="single-kind.s1p",
        )
        candidates = run_tuning_single(
            dut=dut, library=library, port_index=0,
            bands_mhz=[[1000, 3000]], max_components=1,
            topology_filter=["1-Element (Series-L)"],
            beam_width=4, timeout_seconds=5, num_band_points=3,
        )
    assert candidates
    assert all(result.search_diagnostics["numeric_core"] == "rfmatch_core" for result in candidates.values())
    assert all(result.search_diagnostics["available_component_kinds"] == ["L"] for result in candidates.values())
    assert all(result.total_component_count == 1 for result in candidates.values())


@pytest.mark.parametrize("kwargs, message", [
    ({"max_components": -1}, "between zero and six"),
    ({"max_components": 7}, "between zero and six"),
    ({"bands_mhz": []}, "at least one optimization band"),
    ({"bands_mhz": [[2500, 2400]]}, "each optimization band"),
    ({"port_index": 2}, "outside the DUT port range"),
])
def test_single_authoritative_service_rejects_invalid_search_boundaries(kwargs, message):
    dut = parse_touchstone("# GHZ S RI R 50\n2.4 0.2 0.0\n2.5 0.2 0.0\n", filename="dut.s1p")
    parameters = {
        "dut": dut, "library": ComponentLibrary(), "port_index": 0,
        "bands_mhz": [[2400, 2500]], "max_components": 0,
        "num_band_points": 2,
    }
    parameters.update(kwargs)
    with pytest.raises(ValueError, match=message):
        run_tuning_single(**parameters)


def test_single_product_path_propagates_progress_and_cooperative_cancel():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        library = ComponentLibrary()
        for kind, value in (("L", 4.7), ("C", 1.0)):
            path = root / f"{kind}.s2p"
            _write_series_component(
                path, kind, value * (1e-9 if kind == "L" else 1e-12)
            )
            library.add_component(ComponentInfo(
                part_number=path.stem,
                s2p_filename=str(path),
                zip_path="__DIR__",
                component_type="inductor" if kind == "L" else "capacitor",
                nominal_value=value,
                nominal_unit="nH" if kind == "L" else "pF",
            ))
        dut = parse_touchstone(
            "# GHZ S RI R 50\n1.0 0.35 0.15\n2.0 0.35 0.15\n3.0 0.35 0.15\n",
            filename="cancel-single.s1p",
        )
        events = []
        candidates = run_tuning_single(
            dut=dut,
            library=library,
            port_index=0,
            bands_mhz=[[1000.0, 3000.0]],
            max_components=2,
            objective="balanced",
            beam_width=2,
            num_band_points=3,
            timeout_seconds=60,
            progress_callback=events.append,
            cancel_check=lambda: True,
        )

        assert candidates
        diagnostics = candidates[0].search_diagnostics
        assert diagnostics["search_truncated"] is True
        assert "cancelled during" in diagnostics["termination_reason"]
        assert diagnostics["physical_evaluations"] == 1
        assert any(event["stage"] == "per_port" for event in events)


def test_zip_component_library_uses_lazy_six_element_physical_search():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        zip_path = root / "components.zip"
        records = []
        with zipfile.ZipFile(zip_path, "w") as archive:
            for kind, value in (("L", 4.7), ("C", 0.8)):
                path = root / f"{kind}{value}.s2p"
                _write_series_component(
                    path, kind, value * (1e-9 if kind == "L" else 1e-12)
                )
                archive.write(path, arcname=path.name)
                records.append((kind, value, path.name))
        library = ComponentLibrary()
        for kind, value, filename in records:
            library.add_component(ComponentInfo(
                part_number=Path(filename).stem,
                s2p_filename=filename,
                zip_path=str(zip_path),
                component_type="inductor" if kind == "L" else "capacitor",
                nominal_value=value,
                nominal_unit="nH" if kind == "L" else "pF",
            ))
        dut = parse_touchstone(
            "# GHZ S RI R 50\n1.0 0.35 0.15\n2.0 0.35 0.15\n3.0 0.35 0.15\n",
            filename="zip-backed.s1p",
        )

        candidates = run_tuning_single(
            dut=dut,
            library=library,
            port_index=0,
            bands_mhz=[[1000.0, 3000.0]],
            max_components=6,
            objective="balanced",
            beam_width=2,
            num_band_points=3,
        )

        assert candidates
        best = candidates[0]
        assert best.efficiency_basis == "rfmatch_core_physical_measured_s2p"
        assert best.search_diagnostics["component_model_backends"] == ["adapter"]
        assert best.search_diagnostics["maximum_components_searched"] == 6
        assert best.search_diagnostics["component_models_loaded"] > 0
        assert best.search_diagnostics.get("measured_physical_fallback_reason") is None

        started = time.perf_counter()
        bounded = run_tuning_single(
            dut=dut,
            library=library,
            port_index=0,
            bands_mhz=[[1000.0, 3000.0]],
            max_components=6,
            objective="balanced",
            beam_width=2,
            timeout_seconds=0.01,
            num_band_points=3,
        )
        elapsed = time.perf_counter() - started
        assert bounded
        assert bounded[0].search_diagnostics["search_truncated"] is True
        assert bounded[0].search_diagnostics["termination_reason"]
        assert elapsed < 2.0


def test_lazy_zip_adapter_preserves_touchstone_reference_resistance():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "L4.7.s2p"
        archive_path = root / "components.zip"
        _write_series_component(source, "L", 4.7e-9, reference_resistance=75.0)
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.write(source, arcname=source.name)
        library = ComponentLibrary()
        library.add_component(ComponentInfo(
            part_number="L4.7",
            s2p_filename=source.name,
            zip_path=str(archive_path),
            component_type="inductor",
            nominal_value=4.7,
            nominal_unit="nH",
        ))

        catalog = _core_component_catalog(
            library,
            "L",
            np.asarray([1.0e9, 2.0e9, 3.0e9]),
        )

        assert library.inductors[0]._data is None
        model = load_component_model(catalog[0])
        assert model.z0 == pytest.approx(75.0)


def test_product_core_catalog_preserves_distinct_models_at_same_nominal_value():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        library = ComponentLibrary()
        for name, reference_resistance in (("L_A", 50.0), ("L_B", 75.0)):
            path = root / f"{name}.s2p"
            _write_series_component(
                path, "L", 4.7e-9,
                reference_resistance=reference_resistance,
            )
            library.add_component(ComponentInfo(
                part_number=name,
                s2p_filename=str(path),
                zip_path="__DIR__",
                component_type="inductor",
                nominal_value=4.7,
                nominal_unit="nH",
                tolerance_pct=2.0 if name == "L_A" else 10.0,
            ))

        catalog = _core_component_catalog(
            library, "L", np.asarray([1.0e9, 2.0e9, 3.0e9])
        )

    assert [item.name for item in catalog] == ["L_A", "L_B"]
    assert len({item.value for item in catalog}) == 1
    assert [item.tolerance for item in catalog] == pytest.approx([0.02, 0.10])


def test_repository_murata_sqlite_library_uses_bounded_lazy_physical_search():
    root = Path(__file__).resolve().parents[2]
    database = root / "data" / "Murata" / "murata_components.db"
    dut_path = (
        root / "benchmarks" / "optenni_exports"
        / "optimization_settings_original.s1p"
    )
    if not database.exists() or not dut_path.exists():
        pytest.skip("repository Murata SQLite baseline is not available")
    library = load_murata_db(str(database))
    try:
        catalog_size = len(library.inductors) + len(library.capacitors)
        candidates = run_tuning_single(
            dut=load_touchstone_file(dut_path),
            library=library,
            port_index=0,
            bands_mhz=[[1700.0, 2500.0]],
            max_components=2,
            objective="balanced",
            beam_width=2,
            num_band_points=4,
            timeout_seconds=30,
        )
    finally:
        library.close()

    assert candidates
    diagnostics = candidates[0].search_diagnostics
    assert diagnostics["component_model_backends"] == ["adapter"]
    assert diagnostics["topology_code"] == "PCSL"
    assert 0 < diagnostics["component_models_loaded"] < catalog_size / 20
    assert candidates[0].maximum_power_balance_error < 1e-10


def test_single_port_measured_search_preserves_multiport_coupling_power():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        library = ComponentLibrary()
        for kind, value in (("L", 4.7), ("C", 1.0)):
            path = root / f"{kind}.s2p"
            _write_series_component(
                path, kind, value * (1e-9 if kind == "L" else 1e-12)
            )
            library.add_component(ComponentInfo(
                part_number=path.stem,
                s2p_filename=str(path),
                zip_path="__DIR__",
                component_type="inductor" if kind == "L" else "capacitor",
                nominal_value=value,
                nominal_unit="nH" if kind == "L" else "pF",
            ))
        dut = parse_touchstone(
            "# GHZ S RI R 50\n"
            "1.0 0.30 0 0.40 0 0.40 0 0.20 0\n"
            "2.0 0.30 0 0.40 0 0.40 0 0.20 0\n"
            "3.0 0.30 0 0.40 0 0.40 0 0.20 0\n",
            filename="coupled.s2p",
        )

        best = run_tuning_single(
            dut=dut,
            library=library,
            port_index=0,
            bands_mhz=[[1000.0, 3000.0]],
            max_components=1,
            objective="balanced",
            beam_width=1,
            num_band_points=3,
        )[0]

        metrics = best.per_port[0]
        assert metrics.coupling_loss > 0.01
        assert best.avg_coupling_loss > 0.01
        assert metrics.accepted_efficiency == pytest.approx(
            metrics.coupling_loss
            + metrics.component_loss
            + metrics.radiated_efficiency,
            abs=1e-10,
        )
        assert best.maximum_power_balance_error < 1e-10


def test_joint_product_path_uses_full_matrix_measured_search_and_port_limits():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        library = ComponentLibrary()
        for kind, value in (("L", 4.7), ("C", 1.0)):
            path = root / f"{kind}.s2p"
            _write_series_component(
                path, kind, value * (1e-9 if kind == "L" else 1e-12)
            )
            library.add_component(ComponentInfo(
                part_number=path.stem,
                s2p_filename=str(path),
                zip_path="__DIR__",
                component_type="inductor" if kind == "L" else "capacitor",
                nominal_value=value,
                nominal_unit="nH" if kind == "L" else "pF",
            ))
        dut = parse_touchstone(
            "# GHZ S RI R 50\n"
            "1.0 0.30 0 0.35 0 0.35 0 0.25 0\n"
            "2.0 0.30 0 0.35 0 0.35 0 0.25 0\n"
            "3.0 0.30 0 0.35 0 0.35 0 0.25 0\n",
            filename="joint-coupled.s2p",
        )

        candidates = run_tuning_joint(
            dut=dut,
            library=library,
            port_specs=[
                {"port_index": 0, "bands_mhz": [[1000, 3000]], "max_components": 0},
                {"port_index": 1, "bands_mhz": [[1000, 3000]], "max_components": 1},
            ],
            objective="balanced",
            beam_width=2,
            num_band_points=3,
        )

        assert candidates
        best = candidates[0]
        assert best.efficiency_basis == "rfmatch_core_physical_measured_s2p_joint"
        assert best.search_diagnostics["numeric_core"] == "rfmatch_core"
        assert best.search_diagnostics["measured_physical_search"] is True
        assert best.search_diagnostics["search_mode"] == "joint_full_matrix"
        assert best.search_diagnostics["per_port_keep"] >= 8
        assert best.search_diagnostics["maximum_components_by_port"] == {"0": 0, "1": 1}
        assert best.search_diagnostics["calibration_reference"]["status"] == "reference_only_not_request_specific"
        assert best.component_choices[0] == []
        assert len(best.component_choices[1]) <= 1
        assert best.per_port[0].coupling_loss > 0.01
        assert best.per_port[1].coupling_loss > 0.01
        assert best.maximum_power_balance_error < 1e-10

        started = time.perf_counter()
        checkpoint = {}
        timed = run_tuning_joint(
            dut=dut,
            library=library,
            port_specs=[
                {"port_index": 0, "bands_mhz": [[1000, 3000]], "max_components": 1},
                {"port_index": 1, "bands_mhz": [[1000, 3000]], "max_components": 1},
            ],
            objective="balanced",
            beam_width=2,
            num_band_points=3,
            timeout_seconds=0,
            checkpoint_store=checkpoint,
        )
        elapsed = time.perf_counter() - started

        assert timed
        assert timed[0].search_diagnostics["search_truncated"] is True
        assert "cancelled during" in timed[0].search_diagnostics["termination_reason"]
        assert timed[0].maximum_power_balance_error < 1e-10
        assert elapsed < 2.0

        next_checkpoint = {}
        resumed = run_tuning_joint(
            dut=dut,
            library=library,
            port_specs=[
                {"port_index": 0, "bands_mhz": [[1000, 3000]], "max_components": 1},
                {"port_index": 1, "bands_mhz": [[1000, 3000]], "max_components": 1},
            ],
            objective="balanced",
            beam_width=2,
            num_band_points=3,
            timeout_seconds=2,
            search_checkpoint=checkpoint,
            checkpoint_store=next_checkpoint,
            search_profile_timeout_seconds=2,
        )
        assert resumed
        diagnostics = resumed[0].search_diagnostics
        assert diagnostics["checkpoint_reused"] is True
        assert diagnostics["checkpoint_prior_physical_evaluations"] == timed[0].search_diagnostics["physical_evaluations"]
        assert next_checkpoint["optimizer"] is checkpoint["optimizer"]


def test_joint_product_path_enforces_per_port_topology_whitelists():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        library = ComponentLibrary()
        for kind, value in (("L", 4.7), ("C", 1.0)):
            path = root / f"{kind}.s2p"
            _write_series_component(
                path, kind, value * (1e-9 if kind == "L" else 1e-12)
            )
            library.add_component(ComponentInfo(
                part_number=path.stem,
                s2p_filename=str(path),
                zip_path="__DIR__",
                component_type="inductor" if kind == "L" else "capacitor",
                nominal_value=value,
                nominal_unit="nH" if kind == "L" else "pF",
            ))
        dut = parse_touchstone(
            "# GHZ S RI R 50\n"
            "1.0 0.30 0.08 0.10 0 0.10 0 0.25 0.05\n"
            "2.0 0.28 0.06 0.09 0 0.09 0 0.23 0.04\n"
            "3.0 0.26 0.04 0.08 0 0.08 0 0.21 0.03\n",
            filename="joint-topology-constraints.s2p",
        )
        checkpoint = {}
        candidates = run_tuning_joint(
            dut=dut,
            library=library,
            port_specs=[
                {
                    "port_index": 0, "bands_mhz": [[1000, 3000]],
                    "max_components": 1, "allowed_topology_codes": ["sl"],
                },
                {
                    "port_index": 1, "bands_mhz": [[1000, 3000]],
                    "max_components": 1, "allowed_topology_codes": ["PC"],
                },
            ],
            objective="balanced",
            beam_width=4,
            num_band_points=3,
            checkpoint_store=checkpoint,
        )

        resumed = run_tuning_joint(
            dut=dut,
            library=library,
            port_specs=[
                {
                    "port_index": 0, "bands_mhz": [[1000, 3000]],
                    "max_components": 1, "allowed_topology_codes": ["SL"],
                },
                {
                    "port_index": 1, "bands_mhz": [[1000, 3000]],
                    "max_components": 1, "allowed_topology_codes": ["PL"],
                },
            ],
            objective="balanced",
            beam_width=4,
            num_band_points=3,
            search_checkpoint=checkpoint,
        )

    assert candidates
    for result in candidates.values():
        assert [
            (item["connection_type"], item["type"])
            for item in result.per_port[0].components
        ] == [("series", "inductor")]
        assert [
            (item["connection_type"], item["type"])
            for item in result.per_port[1].components
        ] == [("shunt", "capacitor")]
        assert result.search_diagnostics["allowed_topology_codes_by_port"] == {
            "0": ["SL"], "1": ["PC"]
        }
        assert result.search_diagnostics["coupled_ideal_topology_search"] is True
        assert result.search_diagnostics["joint_refine_port_blocks"] is True
    assert resumed
    assert checkpoint["optimizer"].config.allowed_topology_codes_by_port == {
        0: frozenset({"SL"}), 1: frozenset({"PL"})
    }
    for result in resumed.values():
        assert [
            (item["connection_type"], item["type"])
            for item in result.per_port[1].components
        ] == [("shunt", "inductor")]


@pytest.mark.parametrize("raw_codes", [[], ["BAD"], ["SLPC"]])
def test_product_topology_contract_rejects_empty_malformed_or_over_depth_codes(raw_codes):
    with pytest.raises(ValueError):
        normalize_allowed_topology_codes(raw_codes, 1, port_index=0)


def test_joint_single_type_catalog_stays_in_full_matrix_measured_core():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "L2.2.s2p"
        _write_series_component(path, "L", 2.2e-9)
        library = ComponentLibrary()
        library.add_component(ComponentInfo(
            part_number="L2.2", s2p_filename=str(path), zip_path="__DIR__",
            component_type="inductor", nominal_value=2.2, nominal_unit="nH",
        ))
        dut = parse_touchstone(
            "# GHZ S RI R 50\n"
            "1.0 0.30 0.10 0.20 0 0.20 0 0.25 0.05\n"
            "2.0 0.28 0.08 0.20 0 0.20 0 0.23 0.04\n"
            "3.0 0.26 0.06 0.20 0 0.20 0 0.21 0.03\n",
            filename="joint-single-kind.s2p",
        )

        candidates = run_tuning_joint(
            dut=dut,
            library=library,
            port_specs=[
                {"port_index": 0, "bands_mhz": [[1000, 3000]], "max_components": 1},
                {"port_index": 1, "bands_mhz": [[1000, 3000]], "max_components": 1},
            ],
            objective="balanced",
            beam_width=8,
            num_band_points=3,
            isolation_targets=[{
                "source_port": 0, "destination_port": 1,
                "band_mhz": [1000, 3000], "maximum_db": -10.0,
            }],
        )

    assert candidates
    assert any(result.total_component_count > 0 for result in candidates.values())
    for result in candidates.values():
        assert result.search_diagnostics["numeric_core"] == "rfmatch_core"
        assert result.search_diagnostics["available_component_kinds"] == ["L"]
        assert result.search_diagnostics["component_catalog_search_unavailable"] is False
        assert result.maximum_power_balance_error < 1e-10
        assert result.isolation_targets
        assert all(
            component["type"] == "inductor"
            for metrics in result.per_port.values()
            for component in metrics.components
        )


def test_joint_zero_component_baseline_does_not_require_library():
    dut = parse_touchstone(
        "# GHZ S RI R 50\n"
        "1.0 0.30 0 0.15 0 0.15 0 0.20 0\n"
        "2.0 0.25 0 0.15 0 0.15 0 0.18 0\n",
        filename="joint-bare.s2p",
    )
    candidates = run_tuning_joint(
        dut=dut,
        library=None,
        port_specs=[
            {"port_index": 0, "bands_mhz": [[1000, 2000]], "band_weights": [3], "port_weight": 2, "max_components": 0},
            {"port_index": 1, "bands_mhz": [[1000, 2000]], "band_weights": [2], "port_weight": 0.5, "max_components": 0},
        ],
        objective="balanced",
        beam_width=2,
        num_band_points=2,
    )

    assert candidates
    assert len(candidates) == 1
    result = candidates[0]
    assert result.total_component_count == 0
    assert result.efficiency_basis == "rfmatch_core_physical_bare_dut_joint"
    assert result.search_diagnostics["bare_dut_core_baseline"] is True
    assert result.search_diagnostics["measured_physical_search"] is False
    assert result.search_diagnostics["available_component_kinds"] == []
    assert result.search_diagnostics["priority_weights_by_port"] == {
        "0": {
            "port_weight": 2.0, "band_weights": [3.0],
            "effective_band_weights": [6.0],
            "semantics": "band_margin_multiplier",
        },
        "1": {
            "port_weight": 0.5, "band_weights": [2.0],
            "effective_band_weights": [1.0],
            "semantics": "band_margin_multiplier",
        },
    }
    assert result.maximum_power_balance_error < 1e-12


def test_joint_measured_search_is_equivalent_for_ri_ma_db_non_50_ohm_dut():
    matrices = np.asarray([
        [[0.30 + 0.08j, 0.12 - 0.02j], [0.09 + 0.01j, 0.24 - 0.05j]],
        [[0.28 + 0.07j, 0.11 - 0.03j], [0.08 + 0.02j, 0.22 - 0.04j]],
        [[0.26 + 0.06j, 0.10 - 0.02j], [0.07 + 0.01j, 0.20 - 0.03j]],
    ], dtype=complex)
    with tempfile.TemporaryDirectory() as directory:
        component_path = Path(directory) / "L2.2.s2p"
        _write_series_component(component_path, "L", 2.2e-9)
        library = ComponentLibrary()
        library.add_component(ComponentInfo(
            part_number="L2.2", s2p_filename=str(component_path), zip_path="__DIR__",
            component_type="inductor", nominal_value=2.2, nominal_unit="nH",
        ))
        results = {}
        for fmt in ("RI", "MA", "DB"):
            dut = parse_touchstone(
                _serialize_dut(matrices, fmt), filename=f"non-50-{fmt}.s2p"
            )
            assert dut.reference_resistance == 75.0
            results[fmt] = run_tuning_joint(
                dut=dut,
                library=library,
                port_specs=[
                    {"port_index": 0, "bands_mhz": [[1000, 3000]], "max_components": 1},
                    {"port_index": 1, "bands_mhz": [[1000, 3000]], "max_components": 1},
                ],
                objective="balanced", beam_width=8, num_band_points=3,
            )

    reference = results["RI"]
    assert reference
    for fmt in ("MA", "DB"):
        assert len(results[fmt]) == len(reference)
        for index in reference:
            expected, actual = reference[index], results[fmt][index]
            assert actual.system_score == pytest.approx(expected.system_score, abs=2e-12)
            assert actual.total_component_count == expected.total_component_count
            assert actual.search_diagnostics["topology_code"] == expected.search_diagnostics["topology_code"]
            assert actual.maximum_power_balance_error < 1e-10


def test_joint_measured_search_uses_real_per_port_reference_impedances():
    common = np.asarray([
        [[0.30 + 0.08j, 0.12 - 0.02j], [0.09 + 0.01j, 0.24 - 0.05j]],
        [[0.28 + 0.07j, 0.11 - 0.03j], [0.08 + 0.02j, 0.22 - 0.04j]],
        [[0.26 + 0.06j, 0.10 - 0.02j], [0.07 + 0.01j, 0.20 - 0.03j]],
    ], dtype=complex)
    per_port = renormalize_s_parameters(common, 50.0, np.array([50.0, 75.0]))

    def ri_rows(matrices):
        rows = []
        for frequency, matrix in zip((1.0, 2.0, 3.0), matrices):
            fields = [f"{frequency:g}"]
            for source in range(2):
                for destination in range(2):
                    value = matrix[destination, source]
                    fields.extend((f"{value.real:.16g}", f"{value.imag:.16g}"))
            rows.append(" ".join(fields))
        return "\n".join(rows)

    collapsed_dut = parse_touchstone(
        "# GHZ S RI R 50\n" + ri_rows(per_port), filename="collapsed.s2p"
    )
    per_port_dut = parse_touchstone(
        "[Version] 2.0\n[Number of Ports] 2\n[Number of Frequencies] 3\n"
        "[Reference] 50 75\n# GHZ S RI R 50\n[Network Data]\n"
        + ri_rows(per_port) + "\n[End]",
        filename="per-port.ts",
    )
    assert per_port_dut.port_impedances == [50 + 0j, 75 + 0j]

    with tempfile.TemporaryDirectory() as directory:
        component_path = Path(directory) / "L2.2.s2p"
        _write_series_component(component_path, "L", 2.2e-9)
        library = ComponentLibrary()
        library.add_component(ComponentInfo(
            part_number="L2.2", s2p_filename=str(component_path), zip_path="__DIR__",
            component_type="inductor", nominal_value=2.2, nominal_unit="nH",
        ))
        searches = [
            run_tuning_joint(
                dut=dut, library=library,
                port_specs=[
                    {"port_index": 0, "bands_mhz": [[1000, 3000]], "max_components": 1},
                    {"port_index": 1, "bands_mhz": [[1000, 3000]], "max_components": 1},
                ],
                objective="balanced", beam_width=8, num_band_points=3,
            )
            for dut in (collapsed_dut, per_port_dut)
        ]

    collapsed_results, per_port_results = searches
    assert len(collapsed_results) == len(per_port_results)
    collapsed_component = next(
        result for result in collapsed_results.values()
        if result.total_component_count > 0
    )
    per_port_component = next(
        result for result in per_port_results.values()
        if result.total_component_count > 0
    )
    assert per_port_component.system_score != pytest.approx(
        collapsed_component.system_score, abs=1e-5
    )
    bare = next(
        result for result in per_port_results.values()
        if result.total_component_count == 0
    )
    expected_efficiency = 1.0 - np.sum(np.abs(per_port) ** 2, axis=1)
    for port in (0, 1):
        np.testing.assert_allclose(
            bare.per_port[port].band_total_eff,
            expected_efficiency[:, port],
            atol=2e-12,
        )
    for actual in per_port_results.values():
        assert actual.search_diagnostics["reference_impedances_ohm"] == [50.0, 75.0]
        assert actual.maximum_power_balance_error < 1e-10
