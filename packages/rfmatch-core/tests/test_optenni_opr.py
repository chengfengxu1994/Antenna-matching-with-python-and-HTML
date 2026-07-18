from pathlib import Path

from rfmatch_core import parse_optenni_opr


def test_optenni_opr_extracts_embedded_input_settings_and_saved_winner(tmp_path: Path):
    project = tmp_path / "sample.opr"
    project.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<OptenniData Version="4.3" FileVersion="4.3" hasResults="1">
  <ImpedanceConfigurations><ImpedanceConfigurationItem name="Default">
    <ImpedanceData np="1" nfreq="2" refImp="50" fileName="C:/data/dut.s1p">
      <Data>1e9 0.1 0.2\n2e9 0.2 0.3\n</Data>
    </ImpedanceData>
  </ImpedanceConfigurationItem></ImpedanceConfigurations>
  <ResultTree><RootTreeItem>
    <MultiPortTreeItem name="P1:"><MultiPortDialogSave /></MultiPortTreeItem>
    <MultiPortTreeItem name="P1:SCPL">
      <MultiPortDialogSave alphaInBand="0.05" alphaTotal="0.1" initialSearchDeepness="0.2"
        indToler="2" capToler="2" msWidthToler="20" useTLSynthesis="1" includeSubstrate="1">
        <FrequencySetList><MultiPortFrequencySet><FrequencyModelList><FrequencyModel>
          <FrequencyItem sparI="1" startFreq="1e9" endFreq="2e9" startFreqInd="0" endFreqInd="1" label="Band" sparLabel="S11" />
        </FrequencyModel></FrequencyModelList></MultiPortFrequencySet></FrequencySetList>
      </MultiPortDialogSave>
      <Capacitor_from_series orientation="hor" componentLabel="C1"><ComponentFromSeries
        filename="C:/library/C1P0.s2p" code="C1P0" value="1" manufacturer="Vendor"
        seriesName="RF" subdirectory="Capacitors/RF" relTol="-1" absTol="0.1" />
      </Capacitor_from_series>
      <Inductor_from_series orientation="vert" componentLabel="L1"><ComponentFromSeries
        filename="C:/library/L2N0.s2p" code="L2N0" value="2" manufacturer="Vendor"
        seriesName="RF" subdirectory="Inductors/RF" relTol="5" absTol="-1" />
      </Inductor_from_series>
    </MultiPortTreeItem>
  </RootTreeItem></ResultTree>
</OptenniData>""",
        encoding="utf-8",
    )

    result = parse_optenni_opr(project)

    assert result["optenni_version"] == "4.3"
    assert result["has_results"] is True
    assert result["impedance_configuration"] == {
        "name": "Default",
        "ports": 1,
        "frequency_points": 2,
        "reference_impedance_ohm": 50.0,
        "source_filename": "dut.s1p",
        "embedded_data_sha256": result["impedance_configuration"]["embedded_data_sha256"],
        "embedded_data_rows": 2,
    }
    assert result["objective"]["alpha_total"] == 0.1
    assert result["manufacturing_tolerance_settings"] == {
        "source_element": "MultiPortDialogSave",
        "inductor_value_tolerance_pct": 2.0,
        "capacitor_value_tolerance_pct": 2.0,
        "microstrip_width_tolerance_pct": 20.0,
        "uses_transmission_line_synthesis": True,
        "includes_substrate": True,
    }
    assert result["bands"][0]["s_parameter"] == "S11"
    assert result["candidate_count"] == 2
    assert result["matched_candidate_count"] == 1
    winner = result["saved_winner"]
    assert winner["topology_by_port"] == {"0": "SCPL"}
    assert [
        (item["connection"], item["component_type"], item["part_number"], item["value_si"])
        for item in winner["components"]
    ] == [
        ("series", "capacitor", "C1P0", 1e-12),
        ("shunt", "inductor", "L2N0", 2e-9),
    ]
