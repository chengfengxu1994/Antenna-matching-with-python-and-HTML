"""Versioned, self-contained HTML reports for saved RF Matching projects."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from typing import Any


REPORT_SCHEMA_VERSION = 1


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _number(value: Any, digits: int = 3, suffix: str = "") -> str:
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "—"


def _percent(value: Any, digits: int = 1) -> str:
    try:
        return f"{100.0 * float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def _scientific(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}e}"
    except (TypeError, ValueError):
        return "—"


def _candidate_score(solution: dict) -> str:
    basis = str(solution.get("efficiency_basis", ""))
    if basis.startswith("rfmatch_core_physical") or "mdif" in basis:
        return _number(solution.get("system_score"), 3, " dB")
    return _percent(solution.get("system_score"))


def _per_port(solution: dict) -> list[tuple[str, dict]]:
    return sorted(
        ((str(key), value) for key, value in (solution.get("per_port") or {}).items()),
        key=lambda item: int(item[0]),
    )


def _curve_points(metrics: dict, field: str) -> tuple[list[float], list[float]]:
    frequencies = metrics.get("band_freqs_hz") or []
    values = metrics.get(field) or []
    grouped: dict[float, list[float]] = {}
    for frequency, value in zip(frequencies, values):
        grouped.setdefault(float(frequency), []).append(float(value))
    x_values = sorted(grouped)
    y_values = [sum(grouped[x]) / len(grouped[x]) for x in x_values]
    return x_values, y_values


def _svg_chart(metrics: dict, field: str, *, title: str, efficiency: bool = False) -> str:
    x_values, y_values = _curve_points(metrics, field)
    if len(x_values) < 2:
        return f'<div class="chart empty">No stored data for {_escape(title)}</div>'
    if not efficiency:
        y_values = [-abs(value) for value in y_values]
    else:
        y_values = [100.0 * value for value in y_values]
    width, height = 680, 220
    left, right, top, bottom = 54, 16, 24, 34
    x_min, x_max = min(x_values), max(x_values)
    if efficiency:
        y_min, y_max = 0.0, max(100.0, max(y_values))
    else:
        y_min, y_max = min(-30.0, min(y_values)), 0.0
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1e-12)
    positive_steps = [
        right_x - left_x
        for left_x, right_x in zip(x_values, x_values[1:])
        if right_x > left_x
    ]
    typical_step = sorted(positive_steps)[len(positive_steps) // 2]
    segments: list[list[tuple[float, float]]] = [[]]
    for index, pair in enumerate(zip(x_values, y_values)):
        if index and x_values[index] - x_values[index - 1] > typical_step * 4.0:
            segments.append([])
        segments[-1].append(pair)

    def svg_points(segment: list[tuple[float, float]]) -> str:
        return " ".join(
            f"{left + (x - x_min) / x_span * (width-left-right):.2f},"
            f"{top + (y_max-y) / y_span * (height-top-bottom):.2f}"
            for x, y in segment
        )

    polylines = "".join(
        f'<polyline points="{svg_points(segment)}" class="curve {"efficiency" if efficiency else "s11"}"/>'
        for segment in segments
    )
    y_ticks = [0, 25, 50, 75, 100] if efficiency else [0, -10, -20, -30]
    grid = []
    for value in y_ticks:
        if y_min <= value <= y_max:
            y = top + (y_max - value) / y_span * (height - top - bottom)
            grid.append(
                f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" class="grid"/>'
                f'<text x="{left-7}" y="{y+4:.2f}" text-anchor="end">{value:g}</text>'
            )
    return (
        f'<div class="chart"><h3>{_escape(title)}</h3>'
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(title)}">'
        f'{"".join(grid)}<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>'
        f'{polylines}'
        f'<text x="{left}" y="{height-8}">{x_min/1e9:.3f} GHz</text>'
        f'<text x="{width-right}" y="{height-8}" text-anchor="end">{x_max/1e9:.3f} GHz</text>'
        f'</svg></div>'
    )


def _component_rows(solution: dict) -> str:
    rows = []
    for port, metrics in _per_port(solution):
        for index, component in enumerate(metrics.get("components") or []):
            rows.append(
                "<tr>"
                f"<td>Port {int(port)+1}</td><td>{index+1}</td>"
                f"<td>{_escape(component.get('connection_type') or component.get('connection') or '—')}</td>"
                f"<td>{_escape(component.get('type') or component.get('comp_type') or '—')}</td>"
                f"<td>{_escape(component.get('part_number') or component.get('part') or 'ideal')}</td>"
                f"<td>{_escape(component.get('value') or '—')}</td>"
                f"<td>{_escape(_component_procurement_metadata(component))}</td>"
                "</tr>"
            )
    return "".join(rows) or '<tr><td colspan="7">No component data</td></tr>'


def _component_procurement_metadata(component: dict) -> str:
    values = []
    if component.get("manufacturer"):
        values.append(str(component["manufacturer"]))
    if component.get("series") and component.get("series") != "Unclassified":
        values.append(str(component["series"]))
    if component.get("package_code"):
        values.append("package " + str(component["package_code"]))
    if component.get("tolerance_pct") is not None:
        values.append(f"±{component['tolerance_pct']}%")
    manufacturing = component.get("manufacturing_tolerances_pct") or {}
    if manufacturing:
        labels = {
            "trace_width": "W", "physical_length": "L",
            "substrate_height": "H", "relative_permittivity": "er",
        }
        values.append("microstrip " + ", ".join(
            f"{labels.get(name, name)} ±{float(value):g}%"
            for name, value in manufacturing.items()
        ))
    if component.get("voltage_code"):
        values.append("voltage code " + str(component["voltage_code"]))
    if component.get("dielectric"):
        values.append(str(component["dielectric"]) + " (inferred)")
    evidence = (component.get("environment_metadata") or {}).get("evidence_level")
    if component.get("tempco_ppm_per_c") is not None:
        values.append(
            f"tempco {component['tempco_ppm_per_c']} ppm/°C"
            + (f" [{evidence}]" if evidence else "")
        )
    if component.get("systematic_bias_pct") is not None:
        values.append(
            f"systematic bias {component['systematic_bias_pct']}%"
            + (f" [{evidence}]" if evidence else "")
        )
    return "; ".join(values) or "—"


def _dependency_rows(document: dict) -> str:
    dependencies = (document.get("configuration") or {}).get("input_dependencies") or []
    return "".join(
        "<tr>"
        f"<td>{_escape(item.get('role'))}</td>"
        f"<td>{_escape(item.get('filename'))}</td>"
        f"<td>{_escape(item.get('location', '—'))}</td>"
        f"<td>{'flip 1↔2; ' if item.get('reverse_ports') else ''}{_escape(item.get('reference_impedance_mode', 'native'))} Z0</td>"
        f"<td><code>{_escape(item.get('sha256'))}</code></td>"
        f"<td>{_escape(item.get('size_bytes', '—'))}</td>"
        "</tr>"
        for item in dependencies
    ) or '<tr><td colspan="6">No external layout dependencies</td></tr>'


def _state_rows(solution: dict) -> str:
    configurations = solution.get("frequency_configurations") or []
    if configurations:
        return "".join(
            "<tr>"
            f"<td>{_escape(item.get('name'))}</td>"
            f"<td>{_escape(' + '.join(f'{band[0]}–{band[1]} MHz' for band in item.get('bands_mhz', [])))}</td>"
            f"<td>{_escape(item.get('state'))}</td>"
            f"<td>{_number(item.get('score_db'), 3, ' dB')}</td>"
            "</tr>"
            for item in configurations
        )
    states = solution.get("tunable_states") or {}
    return "".join(
        f"<tr><td>{_escape(name)}</td><td>—</td><td>{_escape(state)}</td><td>—</td></tr>"
        for name, state in states.items()
    ) or '<tr><td colspan="4">Not a state-configured solution</td></tr>'


def _yield_analysis(solution: dict) -> dict:
    return (solution.get("search_diagnostics") or {}).get("yield_analysis") or {}


def _priority_weight_status(solution: dict) -> str:
    diagnostics = solution.get("search_diagnostics") or {}
    by_port = diagnostics.get("priority_weights_by_port") or {}
    if by_port:
        return "; ".join(
            f"P{int(port) + 1}: "
            + " / ".join(str(value) for value in values.get("effective_band_weights") or [])
            for port, values in sorted(by_port.items(), key=lambda item: int(item[0]))
        )
    values = diagnostics.get("priority_weights") or {}
    effective = values.get("effective_band_weights") or []
    port = (solution.get("port_indices") or [0])[0]
    return (
        f"P{int(port) + 1}: " + " / ".join(str(value) for value in effective)
        if effective else "default 1.0"
    )


def _synthesis_seed(solution: dict) -> dict:
    return (solution.get("search_diagnostics") or {}).get("loss_aware_ideal_seed") or {}


def _search_diagnostics(solution: dict) -> dict:
    return solution.get("search_diagnostics") or {}


def _component_library_status(solution: dict) -> str:
    metadata = _search_diagnostics(solution).get("component_library_filter") or {}
    if not metadata:
        return "—"
    selected = metadata.get("selected_series")
    if selected is None:
        selection = "default measured catalog"
    else:
        selection = ", ".join(str(item).split("::", 1)[-1] for item in selected) or "none"
    fingerprint = str(metadata.get("catalog_fingerprint") or "")
    suffix = f"; SHA-256 {fingerprint}" if fingerprint else ""
    parameter_filter = metadata.get("parameter_filter") or {}
    constraints = []
    if parameter_filter.get("manufacturers"):
        constraints.append("manufacturers=" + ", ".join(parameter_filter["manufacturers"]))
    if parameter_filter.get("package_codes"):
        constraints.append("packages=" + ", ".join(parameter_filter["package_codes"]))
    if parameter_filter.get("voltage_codes"):
        constraints.append("voltage codes=" + ", ".join(parameter_filter["voltage_codes"]))
    if parameter_filter.get("dielectrics"):
        constraints.append("dielectrics=" + ", ".join(parameter_filter["dielectrics"]))
    if parameter_filter.get("maximum_tolerance_pct") is not None:
        constraints.append(f"tolerance≤{parameter_filter['maximum_tolerance_pct']}%")
    if constraints:
        constraints.append(
            "unknown=" + str(parameter_filter.get("unknown_metadata_policy", "include"))
        )
    filter_suffix = "; " + "; ".join(constraints) if constraints else ""
    return (
        f"{selection}; {metadata.get('inductors', 0)} L / "
        f"{metadata.get('capacitors', 0)} C{filter_suffix}{suffix}"
    )


def _measured_search_status(solution: dict) -> str:
    diagnostics = _search_diagnostics(solution)
    if diagnostics.get("bare_dut_core_baseline"):
        return "full-band bare-DUT physical baseline (rfmatch_core)"
    if diagnostics.get("measured_physical_search"):
        backends = ", ".join(diagnostics.get("component_model_backends") or [])
        prefix = (
            "time-budgeted partial physical"
            if diagnostics.get("search_truncated")
            else "full-band physical"
        )
        reason = diagnostics.get("termination_reason")
        return (
            f"{prefix} ({backends or 'measured model'})"
            + (f": {reason}" if reason else "")
        )
    reason = diagnostics.get("measured_physical_fallback_reason")
    return f"compatibility fallback: {reason}" if reason else "not applicable"


def _search_calibration_status(solution: dict) -> str:
    calibration = _search_diagnostics(solution).get("calibration_reference") or {}
    if not calibration:
        return "—"
    recall = calibration.get("reference_exact_top_k_recall")
    recall_text = _percent(recall) if recall is not None else "—"
    status = (
        f"Reference-only calibration: {recall_text} exact top-k recall; "
        "not proof for this request's catalog"
    )
    if calibration.get("artifact_sha256"):
        status += f"; evidence SHA-256 {calibration['artifact_sha256'][:12]}…"
    golden = calibration.get("numerical_golden") or {}
    if golden:
        status += (
            f"; saved-winner efficiency delta ≤ "
            f"{golden.get('maximum_efficiency_delta_from_rounded_ui_db', '—')} dB"
        )
    discovery = calibration.get("saved_winner_discovery") or {}
    if discovery:
        status += (
            f"; exact saved winner automatically rediscovered at rank "
            f"{discovery.get('exact_saved_winner_rank', '—')} on its reference BOM grid"
        )
    full_discovery = calibration.get("full_catalog_discovery") or {}
    if full_discovery:
        status += (
            f"; full reference catalog retains saved BOM at rank "
            f"{full_discovery.get('exact_saved_winner_rank', '—')}"
        )
    automatic = calibration.get("automatic_full_catalog_discovery") or {}
    if automatic:
        status += (
            f"; unconstrained full-catalog topology rank "
            f"{automatic.get('product_saved_topology_rank', automatic.get('saved_topology_rank', '—'))} and BOM rank "
            f"{automatic.get('product_exact_saved_winner_rank', automatic.get('exact_saved_winner_rank', '—'))}"
        )
    single_product = calibration.get("product_full_catalog_discovery") or {}
    if single_product:
        status += (
            f"; product full-catalog {single_product.get('topology', '—')} "
            f"topology rank {single_product.get('topology_rank', '—')} with max "
            f"efficiency delta {single_product.get('maximum_efficiency_delta_db', '—')} dB"
        )
    four_port = calibration.get("four_port_scaling") or {}
    if four_port:
        four_scope = four_port.get("scope") or {}
        four_recall = four_port.get("reference_exact_top_k_recall")
        status += (
            f"; {four_scope.get('ports', 4)}-port scaling reference "
            f"{_percent(four_recall) if four_recall is not None else '—'} exact top-k recall, "
            f"{four_port.get('heuristic_physical_evaluations', '—')}/"
            f"{four_port.get('exhaustive_physical_evaluations', '—')} physical evaluations"
        )
        if four_port.get("artifact_sha256"):
            status += f" (SHA-256 {four_port['artifact_sha256'][:12]}…)"
    return status


def _search_calibration_scope(solution: dict) -> str:
    scope = (
        _search_diagnostics(solution).get("calibration_reference") or {}
    ).get("scope") or {}
    if not scope:
        return "—"
    return (
        f"{scope.get('ports', '—')} ports · max {scope.get('maximum_components_per_port', '—')} "
        f"component/port · {scope.get('catalog', '—')} · per-port keep "
        f"{scope.get('per_port_keep', '—')} · top-{scope.get('top_k', '—')}"
    )


def _synthesis_seed_topology(solution: dict) -> str:
    signature = _synthesis_seed(solution).get("topology_signature") or []
    return " → ".join(
        f"{str(item[0]).lower()} {str(item[1]).upper()}"
        for item in signature
        if len(item) >= 2
    ) or "—"


def _synthesis_seed_values(solution: dict) -> str:
    elements = _synthesis_seed(solution).get("elements") or []
    values = []
    for item in elements:
        try:
            value = float(item.get("value_si"))
            kind = str(item.get("kind", "")).upper()
            rendered = (
                f"{value * 1e9:.4g} nH" if kind == "L"
                else f"{value * 1e12:.4g} pF"
            )
            values.append(f"{item.get('connection', '—')} {kind} {rendered}")
        except (TypeError, ValueError):
            continue
    return " · ".join(values) or "—"


def _generic_synthesis_loss_status(solution: dict) -> str:
    model = _search_diagnostics(solution).get("generic_synthesis_loss") or {}
    if not model:
        model = _synthesis_seed(solution).get("requested_loss_model") or {}
    if not model:
        return "—"
    try:
        return (
            f"L Q={float(model.get('inductor_q')):g} @ "
            f"{float(model.get('inductor_q_reference_hz')) / 1e9:g} GHz; "
            f"L ESR={float(model.get('inductor_esr_ohm', 0.0)):g} Ω; "
            f"C ESR={float(model.get('capacitor_esr_ohm')):g} Ω; "
            "continuous topology prior only"
        )
    except (TypeError, ValueError):
        return "—"


def _yield_interval(solution: dict) -> str:
    interval = _yield_analysis(solution).get("yield_confidence_interval") or []
    if len(interval) != 2:
        return "—"
    return f"{_percent(interval[0])} – {_percent(interval[1])}"


def _yield_variation_model(solution: dict) -> str:
    analysis = _yield_analysis(solution)
    model = analysis.get("variation_model") or {}
    distribution = analysis.get("distribution", "—")
    correlation = 100.0 * float(model.get("batch_correlation", 0.0))
    parts = [f"{distribution} · batch correlation {correlation:.1f}%"]
    parts.append(
        f"L/C systematic bias {float(model.get('inductor_bias_pct', 0.0)):g}/"
        f"{float(model.get('capacitor_bias_pct', 0.0)):g}%"
    )
    low, high = model.get("temperature_min_c"), model.get("temperature_max_c")
    if low is not None and high is not None:
        parts.append(
            f"temperature {float(low):g}–{float(high):g} °C "
            f"(reference {float(model.get('reference_temperature_c', 25.0)):g} °C)"
        )
        parts.append(
            f"L/C tempco {float(model.get('inductor_tempco_ppm_per_c', 0.0)):g}/"
            f"{float(model.get('capacitor_tempco_ppm_per_c', 0.0)):g} ppm/°C"
        )
    else:
        parts.append("temperature disabled")
    return " · ".join(parts)


def _configuration_yield_rows(solution: dict) -> str:
    yields = _yield_analysis(solution).get("configuration_yield_fraction") or {}
    states = solution.get("tunable_states") or {}
    if not yields:
        return '<tr><td colspan="3">Single-configuration analysis or no state yield saved</td></tr>'
    return "".join(
        "<tr>"
        f"<td>{_escape(name)}</td>"
        f"<td>{_escape(states.get(name, '—'))}</td>"
        f"<td>{_percent(value)}</td>"
        "</tr>"
        for name, value in yields.items()
    )


def _manual_component_summary(components: list[dict]) -> str:
    if not components:
        return "bare DUT"
    labels = []
    for component in components:
        part = component.get("part_number")
        if part:
            labels.append(str(part))
            continue
        kind = {
            "inductor": "L", "capacitor": "C", "resistor": "R",
            "transmission_line": "line", "open_stub": "open stub",
            "short_stub": "short stub",
        }.get(str(component.get("comp_type", "")), component.get("comp_type", "?"))
        value = component.get("value", component.get("electrical_length_deg", "—"))
        placement = "shunt" if component.get("connection_type") == "shunt" else "series"
        labels.append(f"{placement} {kind} {value}")
    return " → ".join(labels)


def _manual_variant_rows(document: dict) -> str:
    workspace = ((document.get("extensions") or {}).get("manual_workspace") or {})
    variants = workspace.get("variants") or []
    if not variants:
        return '<tr><td colspan="8">No frozen manual variants saved</td></tr>'
    return "".join(
        "<tr>"
        f"<td>{_escape(variant.get('name'))}</td>"
        f"<td>P{int(variant.get('input_port', 0)) + 1}</td>"
        f"<td>{_number(float(variant.get('target_frequency_hz', 0)) / 1e6, 3, ' MHz')}</td>"
        f"<td>{_escape(_manual_component_summary(variant.get('components') or []))}</td>"
        f"<td>{_number((variant.get('metrics') or {}).get('return_loss_db'), 3, ' dB')}</td>"
        f"<td>{_number((variant.get('metrics') or {}).get('return_loss_improvement_db'), 3, ' dB')}</td>"
        f"<td>{_number((variant.get('metrics') or {}).get('vswr'), 3)}</td>"
        f"<td>{_number((variant.get('metrics') or {}).get('input_impedance_real'), 2)} "
        f"{'+' if float((variant.get('metrics') or {}).get('input_impedance_imag', 0)) >= 0 else '−'} j"
        f"{_number(abs(float((variant.get('metrics') or {}).get('input_impedance_imag', 0))), 2, ' Ω')}</td>"
        "</tr>"
        for variant in variants
    )


def render_project_report(document: dict[str, Any]) -> str:
    """Render a validated project document without external assets or scripts."""
    candidates = document.get("results", {}).get("candidates") or []
    selected_index = int(document.get("results", {}).get("selected_index") or 0)
    selected_index = min(max(selected_index, 0), max(len(candidates) - 1, 0))
    selected = candidates[selected_index] if candidates else {}
    input_data = document.get("input") or {}
    input_provenance = input_data.get("provenance") or {}
    software = document.get("software") or {}
    request = (document.get("configuration") or {}).get("tuning_request") or {}
    digest = (document.get("integrity") or {}).get("digest", "")
    migration_history = document.get("migration_history") or []
    migrated_from = migration_history[0].get("from_version") if migration_history else None
    project_schema = f"v{document.get('schema_version', '?')}"
    if migrated_from is not None:
        project_schema += f" (migrated from v{migrated_from})"
    generated_at = datetime.now(timezone.utc).isoformat()

    candidate_rows = "".join(
        ("<tr class=\"selected\">" if index == selected_index else "<tr>")
        + f"<td>{index+1}{' (selected)' if index == selected_index else ''}</td>"
        f"<td>{_candidate_score(item)}</td>"
        f"<td>{_percent(item.get('avg_total_efficiency'))}</td>"
        f"<td>{_percent(item.get('min_total_efficiency'))}</td>"
        f"<td>{_percent(item.get('total_component_loss'), 2)}</td>"
        f"<td>{_escape(item.get('total_component_count', '—'))}</td>"
        f"<td>{_percent(_yield_analysis(item).get('yield_fraction'))}</td>"
        f"<td>{_escape(_yield_interval(item))}</td></tr>"
        for index, item in enumerate(candidates)
    ) or '<tr><td colspan="8">No optimization candidates saved</td></tr>'

    charts = "".join(
        _svg_chart(metrics, "band_s11_db", title=f"Port {int(port)+1} return loss")
        + _svg_chart(metrics, "band_total_eff", title=f"Port {int(port)+1} total efficiency", efficiency=True)
        for port, metrics in _per_port(selected)
    )
    request_json = html.escape(json.dumps(request, ensure_ascii=False, indent=2))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{_escape(document.get('name'))} — RF Matching Report</title>
<style>
:root{{--ink:#17202a;--muted:#667085;--line:#d0d5dd;--accent:#1769aa;--good:#067647}}
*{{box-sizing:border-box}} body{{font:14px/1.45 Arial,sans-serif;color:var(--ink);margin:0;background:#f4f6f8}}
main{{max-width:1120px;margin:24px auto;background:white;padding:34px 42px;box-shadow:0 2px 12px #0002}}
h1{{margin:0;font-size:28px}} h2{{margin-top:30px;border-bottom:2px solid var(--accent);padding-bottom:6px;font-size:18px}}
h3{{font-size:14px;margin:0 0 8px}} .subtitle,.footnote{{color:var(--muted)}}
.meta,.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px 24px}} .meta div,.metric{{padding:9px 0;border-bottom:1px solid #eee}}
.label{{display:block;color:var(--muted);font-size:11px;text-transform:uppercase}} .value{{font-weight:600;word-break:break-word}}
table{{width:100%;border-collapse:collapse;margin-top:8px}} th,td{{padding:8px;border:1px solid var(--line);text-align:left;font-size:12px}} th{{background:#f2f4f7}} tr.selected{{background:#ecfdf3}}
.charts{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} .chart{{border:1px solid var(--line);padding:12px;break-inside:avoid}} svg text{{font-size:10px;fill:var(--muted)}}
.grid{{stroke:#e5e7eb;stroke-width:1}} .axis{{stroke:#667085}} .curve{{fill:none;stroke-width:2}} .curve.s11{{stroke:#d92d20}} .curve.efficiency{{stroke:#079455}}
pre{{white-space:pre-wrap;background:#f8fafc;border:1px solid var(--line);padding:12px;font-size:11px}} code{{font-family:Consolas,monospace}}
@media print{{body{{background:white}} main{{margin:0;max-width:none;box-shadow:none;padding:20px}} .charts{{grid-template-columns:1fr 1fr}} h2{{break-after:avoid}}}}
</style></head><body><main>
<h1>{_escape(document.get('name'))}</h1>
<p class="subtitle">RF Matching traceability report · schema {REPORT_SCHEMA_VERSION} · generated {_escape(generated_at)}</p>
<section class="meta">
<div><span class="label">Input</span><span class="value">{_escape(input_data.get('filename'))}</span></div>
<div><span class="label">Input source</span><span class="value">{_escape(input_provenance.get('source') or 'Unrecorded')} · {_escape(input_provenance.get('ingestion_method') or '—')}</span></div>
<div><span class="label">Source observed</span><span class="value">{_escape(input_provenance.get('observed_at') or '—')}</span></div>
<div><span class="label">Touchstone SHA-256</span><span class="value"><code>{_escape(input_data.get('sha256'))}</code></span></div>
<div><span class="label">Ports / points</span><span class="value">{_escape(input_data.get('num_ports'))} / {_escape(input_data.get('frequency_count'))}</span></div>
<div><span class="label">Frequency range</span><span class="value">{_number(float(input_data.get('frequency_min_hz',0))/1e6,3,' MHz')} – {_number(float(input_data.get('frequency_max_hz',0))/1e6,3,' MHz')}</span></div>
<div><span class="label">Software</span><span class="value">{_escape(software.get('application'))} API {_escape(software.get('api_version'))} / core {_escape(software.get('rfmatch_core_version'))}</span></div>
<div><span class="label">Project schema</span><span class="value">{_escape(project_schema)}</span></div>
<div><span class="label">Project integrity</span><span class="value"><code>{_escape(digest)}</code></span></div>
</section>
<h2>Input dependencies</h2><table><thead><tr><th>Role</th><th>Filename</th><th>Location</th><th>Transform</th><th>SHA-256</th><th>Bytes</th></tr></thead><tbody>{_dependency_rows(document)}</tbody></table>
<h2>Candidate comparison</h2><table><thead><tr><th>Candidate</th><th>Score</th><th>Average η</th><th>Minimum η</th><th>Component loss</th><th>BOM count</th><th>Yield</th><th>Confidence interval</th></tr></thead><tbody>{candidate_rows}</tbody></table>
<h2>Frozen manual variants</h2><table><thead><tr><th>Name</th><th>Port</th><th>Frequency</th><th>Network</th><th>Return loss</th><th>Improvement</th><th>VSWR</th><th>Input impedance</th></tr></thead><tbody>{_manual_variant_rows(document)}</tbody></table>
<h2>Selected solution</h2><section class="metrics">
<div class="metric"><span class="label">Mode / objective</span><span class="value">{_escape(selected.get('mode'))} / {_escape(selected.get('objective'))}</span></div>
<div class="metric"><span class="label">Average total efficiency</span><span class="value">{_percent(selected.get('avg_total_efficiency'))}</span></div>
<div class="metric"><span class="label">Minimum total efficiency</span><span class="value">{_percent(selected.get('min_total_efficiency'))}</span></div>
<div class="metric"><span class="label">Component loss</span><span class="value">{_percent(selected.get('total_component_loss'),2)}</span></div>
<div class="metric"><span class="label">Power balance error</span><span class="value">{_scientific(selected.get('maximum_power_balance_error'))}</span></div>
<div class="metric"><span class="label">Efficiency basis</span><span class="value">{_escape(selected.get('efficiency_basis'))}</span></div>
<div class="metric"><span class="label">Effective port / band priorities</span><span class="value">{_escape(_priority_weight_status(selected))}</span></div>
<div class="metric"><span class="label">Manufacturing yield</span><span class="value">{_percent(_yield_analysis(selected).get('yield_fraction'))}</span></div>
<div class="metric"><span class="label">Yield confidence interval</span><span class="value">{_escape(_yield_interval(selected))}</span></div>
<div class="metric"><span class="label">P5 score margin</span><span class="value">{_number((_yield_analysis(selected).get('score_percentiles_db') or {}).get('5'), 3, ' dB')}</span></div>
<div class="metric"><span class="label">Yield analysis scope</span><span class="value">{_escape(_yield_analysis(selected).get('analysis_scope', 'fixed network'))}</span></div>
<div class="metric"><span class="label">Yield variation model</span><span class="value">{_escape(_yield_variation_model(selected))}</span></div>
<div class="metric"><span class="label">Loss-aware ideal topology seed</span><span class="value">{_escape(_synthesis_seed_topology(selected))}</span></div>
<div class="metric"><span class="label">Ideal seed values</span><span class="value">{_escape(_synthesis_seed_values(selected))}</span></div>
<div class="metric"><span class="label">Ideal seed score / evaluations</span><span class="value">{_number(_synthesis_seed(selected).get('score_db'), 3, ' dB')} / {_escape(_synthesis_seed(selected).get('evaluations', '—'))}</span></div>
<div class="metric"><span class="label">Generic synthesis loss prior</span><span class="value">{_escape(_generic_synthesis_loss_status(selected))}</span></div>
<div class="metric"><span class="label">Measured search path</span><span class="value">{_escape(_measured_search_status(selected))}</span></div>
<div class="metric"><span class="label">Component catalog selection</span><span class="value">{_escape(_component_library_status(selected))}</span></div>
<div class="metric"><span class="label">Physical evaluations / loaded models</span><span class="value">{_escape(_search_diagnostics(selected).get('physical_evaluations', '—'))} / {_escape(_search_diagnostics(selected).get('component_models_loaded', '—'))}</span></div>
<div class="metric"><span class="label">Active physical frequency points</span><span class="value">{_escape(_search_diagnostics(selected).get('active_frequency_points', '—'))}</span></div>
<div class="metric"><span class="label">Requested search quality</span><span class="value">{_escape((_search_diagnostics(selected).get('search_plan') or {}).get('label', _search_diagnostics(selected).get('search_quality_requested', 'auto')))}</span></div>
<div class="metric"><span class="label">Execution strategy</span><span class="value">{_escape((_search_diagnostics(selected).get('search_plan') or {}).get('strategy', 'hierarchical_measured'))}</span></div>
<div class="metric"><span class="label">Search budget</span><span class="value">{_number((_search_diagnostics(selected).get('search_plan') or {}).get('budget_seconds'), 1, ' s')}</span></div>
<div class="metric"><span class="label">Search calibration</span><span class="value">{_escape(_search_calibration_status(selected))}</span></div>
<div class="metric"><span class="label">Calibration scope</span><span class="value">{_escape(_search_calibration_scope(selected))}</span></div>
</section>
<h2>Bill of materials and topology</h2><table><thead><tr><th>Port</th><th>Position</th><th>Connection</th><th>Type</th><th>Part number</th><th>Value</th><th>Procurement metadata</th></tr></thead><tbody>{_component_rows(selected)}</tbody></table>
<h2>Configuration / state assignment</h2><table><thead><tr><th>Configuration</th><th>Active bands</th><th>State</th><th>Score</th></tr></thead><tbody>{_state_rows(selected)}</tbody></table>
<h2>Configuration yield</h2><table><thead><tr><th>Configuration</th><th>State</th><th>Individual yield</th></tr></thead><tbody>{_configuration_yield_rows(selected)}</tbody></table>
<h2>Stored full-physical curves</h2><div class="charts">{charts or '<p>No stored curve data.</p>'}</div>
<h2>Optimization configuration</h2><pre>{request_json}</pre>
<p class="footnote">This self-contained report is derived from an integrity-checked project snapshot. Exact recomputation also requires the original Touchstone and component model files identified by the project metadata.</p>
</main></body></html>"""
