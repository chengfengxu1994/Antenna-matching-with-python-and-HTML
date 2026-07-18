import os
import sys


sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), "..", "..", "apps", "api"
))

from api.models import MultiScenarioOptimizeRequest, TuningOptimizeRequest
from engine.search_quality import build_multi_scenario_search_plan, build_search_plan


def _ports(constrained=False):
    codes = [["SCPL"], ["PCSL"], ["PCSL"]] if constrained else [None] * 3
    return [
        {
            "port_index": index,
            "bands_mhz": [[1000, 1100]],
            "max_components": 2,
            "allowed_topology_codes": codes[index],
            "enabled": True,
        }
        for index in range(3)
    ]


def test_named_quality_profile_is_authoritative():
    request = TuningOptimizeRequest(
        mode="joint", ports=_ports(), search_quality="quick",
        timeout_seconds=999, beam_width=99, num_band_points=20,
    )
    assert request.timeout_seconds == 15.0
    assert request.beam_width == 8
    assert request.num_band_points == 3


def test_custom_quality_preserves_engineering_limits():
    request = TuningOptimizeRequest(
        mode="joint", ports=_ports(), search_quality="custom",
        timeout_seconds=77, beam_width=17, num_band_points=7,
    )
    assert (request.timeout_seconds, request.beam_width, request.num_band_points) == (
        77, 17, 7
    )


def test_search_plan_truthfully_reports_automatic_topology_eligibility():
    request = TuningOptimizeRequest(
        mode="joint", ports=_ports(), search_quality="exhaustive"
    )
    plan = build_search_plan(request.model_dump())
    assert plan["effective_quality"] == "exhaustive"
    assert plan["strategy"] == "automatic_topology_deep"
    assert plan["automatic_topology_eligible"] is True
    assert plan["budget_seconds"] == 150.0


def test_search_plan_distinguishes_constrained_coupled_strategy():
    request = TuningOptimizeRequest(
        mode="joint", ports=_ports(constrained=True), search_quality="thorough"
    )
    plan = build_search_plan(request.model_dump())
    assert plan["strategy"] == "constrained_coupled_thorough"
    assert plan["constrained_coupled_eligible"] is True
    assert plan["automatic_topology_eligible"] is False


def test_multi_scenario_named_profile_and_plan_share_product_semantics():
    request = MultiScenarioOptimizeRequest(
        scenarios=[{"snp_filename": "free.s1p"}, {"snp_filename": "cover.s1p"}],
        topology_names=["2-Element (Series-C, Shunt-L)"],
        search_quality="balanced", timeout_seconds=999, beam_width=99,
        num_band_points=20,
    )
    assert (request.timeout_seconds, request.beam_width, request.num_band_points) == (
        45.0, 10, 5
    )
    plan = build_multi_scenario_search_plan(request.model_dump())
    assert plan["strategy"] == "shared_network_measured_beam"
    assert plan["scenario_count"] == 2
    assert plan["topology_count"] == 1
    assert "not a proof" in plan["expectation"]
