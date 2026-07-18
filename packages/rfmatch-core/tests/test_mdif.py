from pathlib import Path

import numpy as np
import pytest

from rfmatch_core.mdif import load_mdif


def _write_mdif(path: Path, body: str) -> Path:
    path.write_text(body.strip() + "\n", encoding="utf-8")
    return path


def test_load_mdif_preserves_touchstone_port_order_and_interpolates(tmp_path: Path):
    path = _write_mdif(
        tmp_path / "switch.mdif",
        """
!<ComponentMetaData nPorts="2"><ComponentName>Test tuner</ComponentName></ComponentMetaData>
VAR state = 1.0 pF
BEGIN ACDATA
# GHZ S RI R 50
1 0.1 0 0.2 0 0.3 0 0.4 0
2 0.2 0 0.4 0 0.6 0 0.8 0
END
VAR state = 2.0 pF
BEGIN ACDATA
# GHZ S RI R 50
1 0 0 1 0 0 0 0 0
2 0 0 1 0 0 0 0 0
END
""",
    )
    model = load_mdif(path)
    assert model.name == "Test tuner"
    assert [state.value for state in model.states] == [1.0, 2.0]
    state = model.state("1 pF")
    np.testing.assert_allclose(state.frequencies_hz, [1e9, 2e9])
    # File order is S11,S21,S12,S22: matrix row/column orientation must survive.
    np.testing.assert_allclose(state.s_parameters[0], [[0.1, 0.3], [0.2, 0.4]])
    np.testing.assert_allclose(state.at(1.5e9), [[0.15, 0.45], [0.3, 0.6]])
    assert model.state(2.0).value == 2.0


def test_load_mdif_decodes_db_and_rejects_duplicate_states(tmp_path: Path):
    block = """
VAR state = 1 pF
BEGIN ACDATA
# MHZ S DB R 75
100 -6 0 -12 90 -18 -90 -24 180
END
"""
    model = load_mdif(_write_mdif(tmp_path / "db.mdif", block))
    state = model.states[0]
    assert state.z0 == 75.0
    assert state.frequencies_hz[0] == 100e6
    assert abs(abs(state.s_parameters[0, 0, 0]) - 10 ** (-6 / 20)) < 1e-12
    with pytest.raises(ValueError, match="duplicate MDIF state"):
        load_mdif(_write_mdif(tmp_path / "duplicate.mdif", block + block))


def test_load_mdif_rejects_non_monotonic_or_malformed_data(tmp_path: Path):
    text = """
VAR state = 1 pF
BEGIN ACDATA
# HZ S RI R 50
2 0 0 0 0 0 0 0 0
1 0 0 0 0 0 0 0 0
END
"""
    with pytest.raises(ValueError, match="strictly increasing"):
        load_mdif(_write_mdif(tmp_path / "bad.mdif", text))


def test_load_mdif_supports_categorical_multiport_states_and_wrapped_records(tmp_path: Path):
    path = _write_mdif(
        tmp_path / "sp2t.mdif",
        """
!<ComponentMetaData isSwitch="1" nPorts="3"><ComponentName>Test SP2T</ComponentName></ComponentMetaData>
VAR state = all off
BEGIN ACDATA
# GHZ S RI R 50
1 0.1 0 0.2 0 0.3 0
  0.4 0 0.5 0 0.6 0
  0.7 0 0.8 0 0.9 0
2 0.2 0 0.3 0 0.4 0
  0.5 0 0.6 0 0.7 0
  0.8 0 0.9 0 1.0 0
END
VAR state = 01
BEGIN ACDATA
# GHZ S RI R 50
1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
END
""",
    )
    model = load_mdif(path)
    assert [state.label for state in model.states] == ["all off", "01"]
    state = model.state("all off")
    assert state.n_ports == 3
    assert state.s_parameters.shape == (2, 3, 3)
    np.testing.assert_allclose(state.s_parameters[0], [[0.1, 0.4, 0.7], [0.2, 0.5, 0.8], [0.3, 0.6, 0.9]])
    np.testing.assert_allclose(state.at(1.5e9)[0], [0.15, 0.45, 0.75])
    with pytest.raises(ValueError, match="3 ports"):
        state.as_s2p_model()


def test_official_optenni_switch_contract_when_available():
    root = Path(
        r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials\10 - Tunable antennas\10.6 Impedance tuning using a switch"
    )
    sp3t = root / "SP3T_ideal.mdif"
    sp2t = root / "tutorial_SP2T.mdif"
    if not sp3t.exists() or not sp2t.exists():
        pytest.skip("Optenni switch tutorial data is not installed")
    model3 = load_mdif(sp3t)
    model2 = load_mdif(sp2t)
    assert model3.metadata["commonPort"] == "1"
    assert [state.label for state in model3.states] == ["000", "001", "010", "011", "100", "101", "110", "111"]
    assert all(state.s_parameters.shape == (1001, 4, 4) for state in model3.states)
    assert [state.label for state in model2.states] == ["all off", "RFC-RF2", "RFC-RF1", "all on"]
    assert all(state.s_parameters.shape == (1001, 3, 3) for state in model2.states)


def test_official_optenni_variable_capacitor_contract_when_available():
    path = Path(
        r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials\10 - Tunable antennas\10.5 Impedance tuning using a variable capacitor\Variable_capacitor_tutorial.mdif"
    )
    if not path.exists():
        pytest.skip("Optenni tutorial data is not installed")
    model = load_mdif(path)
    assert model.name == "Variable capacitor"
    assert [state.value for state in model.states] == [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]
    assert all(state.unit == "pF" for state in model.states)
    assert all(len(state.frequencies_hz) == 991 for state in model.states)
    assert all(state.s_parameters.shape == (991, 2, 2) for state in model.states)
    assert model.state("8 pF").frequencies_hz[0] == 100e6
