from pathlib import Path

import numpy as np
import pytest

from rfmatch_core import (
    Touchstone,
    compare_optenni_native_plot,
    load_optenni_plot_export,
    load_touchstone,
    replay_one_port_dut_through_network,
)


ROOT = Path(__file__).resolve().parents[3]
EXPORTS = ROOT / "benchmarks" / "optenni_exports"
DUT = Path(
    r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials"
    r"\1 - START HERE\measured_antenna.s1p"
)


def test_native_plot_parser_rejects_duplicate_tolerance_columns(tmp_path):
    export = tmp_path / "duplicate.txt"
    export.write_text(
        '"Frequency [GHz]"\t"S11"\t"S11"\t"Total efficiency"\n'
        '1.0\t-10\t-11\t-1\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_optenni_plot_export(export)


def test_native_network_replay_for_an_ideal_through():
    frequencies = np.asarray([1e9, 2e9])
    dut_gamma = np.asarray([0.2 + 0.1j, -0.1 + 0.3j])
    dut = Touchstone(
        frequencies,
        dut_gamma.reshape(-1, 1, 1),
        50.0,
    )
    through = np.zeros((2, 2, 2), dtype=complex)
    through[:, 0, 1] = 1.0
    through[:, 1, 0] = 1.0
    network = Touchstone(frequencies, through, 50.0)
    replay = replay_one_port_dut_through_network(dut, network)
    np.testing.assert_allclose(replay.input_reflection, dut_gamma)
    np.testing.assert_allclose(
        replay.total_efficiency, 1.0 - np.abs(dut_gamma) ** 2
    )


def test_quick_start_native_export_replays_every_optenni_point():
    if not DUT.is_file():
        pytest.skip("licensed Optenni Quick Start DUT is not installed")
    plot = load_optenni_plot_export(
        EXPORTS / "quick_start_0402cs_gjm15_pcsl_plot.txt"
    )
    replay = replay_one_port_dut_through_network(
        load_touchstone(DUT),
        load_touchstone(
            EXPORTS / "quick_start_0402cs_gjm15_pcsl_circuit.s2p"
        ),
        source_port=0,
        load_port=1,
    )
    comparison = compare_optenni_native_plot(plot, replay)
    assert comparison.points == 531
    assert comparison.maximum_s11_error_db <= 5e-4
    assert comparison.maximum_efficiency_error_db <= 1e-4
