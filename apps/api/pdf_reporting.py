"""Native, paginated PDF traceability reports for validated project snapshots."""

from __future__ import annotations

from io import BytesIO
import html
import json
from pathlib import Path
from typing import Any, Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from reporting import (
    REPORT_SCHEMA_VERSION,
    _candidate_score,
    _component_library_status,
    _component_procurement_metadata,
    _curve_points,
    _measured_search_status,
    _number,
    _per_port,
    _percent,
    _priority_weight_status,
    _scientific,
    _search_calibration_scope,
    _search_calibration_status,
    _search_diagnostics,
    _generic_synthesis_loss_status,
    _synthesis_seed,
    _synthesis_seed_topology,
    _synthesis_seed_values,
    _yield_analysis,
    _yield_interval,
    _yield_variation_model,
)


PDF_REPORT_SCHEMA_VERSION = 1
_ACCENT = colors.HexColor("#1769AA")
_MUTED = colors.HexColor("#667085")
_LINE = colors.HexColor("#D0D5DD")
_LIGHT = colors.HexColor("#F2F4F7")
_SELECTED = colors.HexColor("#ECFDF3")


def _register_fonts() -> tuple[str, str]:
    """Prefer a CJK-capable local font and retain a portable PDF fallback."""
    registered = set(pdfmetrics.getRegisteredFontNames())
    regular_name, bold_name = "RFMatchSans", "RFMatchSansBold"
    if regular_name in registered and bold_name in registered:
        return regular_name, bold_name
    candidates = (
        (
            Path(r"C:\Windows\Fonts\msyh.ttc"),
            Path(r"C:\Windows\Fonts\msyhbd.ttc"),
        ),
        (
            Path(r"C:\Windows\Fonts\arial.ttf"),
            Path(r"C:\Windows\Fonts\arialbd.ttf"),
        ),
    )
    for regular, bold in candidates:
        if not regular.is_file() or not bold.is_file():
            continue
        try:
            pdfmetrics.registerFont(TTFont(regular_name, str(regular), subfontIndex=0))
            pdfmetrics.registerFont(TTFont(bold_name, str(bold), subfontIndex=0))
            pdfmetrics.registerFontFamily(
                "RFMatchSans",
                normal=regular_name,
                bold=bold_name,
                italic=regular_name,
                boldItalic=bold_name,
            )
            return regular_name, bold_name
        except Exception:
            continue
    return "Helvetica", "Helvetica-Bold"


def _ascii_missing(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return str(value).replace("—", "-").replace("–", "-").replace("→", "->")


def _escape(value: Any) -> str:
    return html.escape(_ascii_missing(value)).replace("\n", "<br/>")


def _styles(regular: str, bold: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "RFTitle", parent=base["Title"], fontName=bold, fontSize=22,
            leading=27, textColor=colors.HexColor("#17202A"), alignment=TA_LEFT,
            spaceAfter=4 * mm,
        ),
        "subtitle": ParagraphStyle(
            "RFSubtitle", parent=base["Normal"], fontName=regular, fontSize=8.5,
            leading=12, textColor=_MUTED, spaceAfter=5 * mm,
        ),
        "section": ParagraphStyle(
            "RFSection", parent=base["Heading2"], fontName=bold, fontSize=13,
            leading=16, textColor=_ACCENT, spaceBefore=5 * mm, spaceAfter=2.5 * mm,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "RFBody", parent=base["BodyText"], fontName=regular, fontSize=8.2,
            leading=11, textColor=colors.HexColor("#17202A"), wordWrap="CJK",
        ),
        "small": ParagraphStyle(
            "RFSmall", parent=base["BodyText"], fontName=regular, fontSize=6.8,
            leading=9, textColor=colors.HexColor("#344054"), wordWrap="CJK",
        ),
        "label": ParagraphStyle(
            "RFLabel", parent=base["BodyText"], fontName=bold, fontSize=7,
            leading=9, textColor=_MUTED, wordWrap="CJK",
        ),
        "center": ParagraphStyle(
            "RFCenter", parent=base["BodyText"], fontName=regular, fontSize=7,
            leading=9, alignment=TA_CENTER, wordWrap="CJK",
        ),
        "code": ParagraphStyle(
            "RFCode", parent=base["Code"], fontName=regular, fontSize=6.5,
            leading=8.4, leftIndent=0, wordWrap="CJK",
        ),
    }


def _p(value: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(_escape(value), style)


def _table(
    rows: list[list[Any]],
    widths: list[float],
    styles: dict[str, ParagraphStyle],
    *,
    header: bool = True,
    selected_row: int | None = None,
) -> Table:
    body = [
        [cell if isinstance(cell, Flowable) else _p(cell, styles["small"]) for cell in row]
        for row in rows
    ]
    table = Table(body, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands: list[tuple] = [
        ("GRID", (0, 0), (-1, -1), 0.35, _LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header and rows:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), _LIGHT),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#344054")),
        ])
        body[0] = [_p(cell, styles["label"]) if not isinstance(cell, Flowable) else cell for cell in rows[0]]
    if selected_row is not None:
        commands.append(("BACKGROUND", (0, selected_row), (-1, selected_row), _SELECTED))
    table.setStyle(TableStyle(commands))
    return table


class CurveChart(Flowable):
    """Small vector chart that preserves gaps between disjoint RF bands."""

    def __init__(self, metrics: dict, field: str, title: str, efficiency: bool):
        super().__init__()
        self.metrics = metrics
        self.field = field
        self.title = title
        self.efficiency = efficiency
        self.width = 170 * mm
        self.height = 54 * mm

    def wrap(self, avail_width, avail_height):
        self.width = min(self.width, avail_width)
        return self.width, self.height

    def draw(self):
        canvas = self.canv
        x_values, y_values = _curve_points(self.metrics, self.field)
        canvas.saveState()
        canvas.setFont(self._font_name, 8)
        canvas.setFillColor(colors.HexColor("#17202A"))
        canvas.drawString(0, self.height - 10, self.title)
        if len(x_values) < 2:
            canvas.setFillColor(_MUTED)
            canvas.drawString(0, self.height / 2, "No stored curve data")
            canvas.restoreState()
            return
        if self.efficiency:
            y_values = [100.0 * float(value) for value in y_values]
            y_min, y_max = 0.0, max(100.0, max(y_values))
            ticks = (0.0, 25.0, 50.0, 75.0, 100.0)
            curve_color = colors.HexColor("#079455")
        else:
            y_values = [-abs(float(value)) for value in y_values]
            y_min, y_max = min(-30.0, min(y_values)), 0.0
            ticks = (0.0, -10.0, -20.0, -30.0)
            curve_color = colors.HexColor("#D92D20")
        left, right, bottom, top = 31.0, 8.0, 19.0, 19.0
        plot_width = self.width - left - right
        plot_height = self.height - bottom - top
        x_min, x_max = min(x_values), max(x_values)
        x_span = max(x_max - x_min, 1.0)
        y_span = max(y_max - y_min, 1e-12)

        def point(index: int) -> tuple[float, float]:
            return (
                left + (x_values[index] - x_min) / x_span * plot_width,
                bottom + (y_values[index] - y_min) / y_span * plot_height,
            )

        canvas.setFont(self._font_name, 6.5)
        for tick in ticks:
            if y_min <= tick <= y_max:
                y = bottom + (tick - y_min) / y_span * plot_height
                canvas.setStrokeColor(colors.HexColor("#E5E7EB"))
                canvas.line(left, y, left + plot_width, y)
                canvas.setFillColor(_MUTED)
                canvas.drawRightString(left - 4, y - 2, f"{tick:g}")
        canvas.setStrokeColor(_MUTED)
        canvas.line(left, bottom, left + plot_width, bottom)
        canvas.setFillColor(_MUTED)
        canvas.drawString(left, 4, f"{x_min / 1e9:.3f} GHz")
        canvas.drawRightString(left + plot_width, 4, f"{x_max / 1e9:.3f} GHz")

        steps = [b - a for a, b in zip(x_values, x_values[1:]) if b > a]
        typical_step = sorted(steps)[len(steps) // 2] if steps else x_span
        canvas.setStrokeColor(curve_color)
        canvas.setLineWidth(1.4)
        for index in range(1, len(x_values)):
            if x_values[index] - x_values[index - 1] <= typical_step * 4.0:
                canvas.line(*point(index - 1), *point(index))
        canvas.restoreState()

    _font_name = "Helvetica"


def _selected_solution(document: dict[str, Any]) -> tuple[list[dict], int, dict]:
    candidates = document.get("results", {}).get("candidates") or []
    index = int(document.get("results", {}).get("selected_index") or 0)
    index = min(max(index, 0), max(len(candidates) - 1, 0))
    return candidates, index, candidates[index] if candidates else {}


def _section(title: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return _p(title, styles["section"])


def _metadata_rows(document: dict[str, Any], styles: dict[str, ParagraphStyle]) -> list[list[Any]]:
    input_data = document.get("input") or {}
    input_provenance = input_data.get("provenance") or {}
    software = document.get("software") or {}
    history = document.get("migration_history") or []
    schema = f"v{document.get('schema_version', '?')}"
    if history:
        schema += f" (migrated from v{history[0].get('from_version')})"
    frequency_min = float(input_data.get("frequency_min_hz") or 0.0) / 1e6
    frequency_max = float(input_data.get("frequency_max_hz") or 0.0) / 1e6
    pairs = [
        ("Input", input_data.get("filename")),
        ("Input source", f"{input_provenance.get('source') or 'Unrecorded'} / {input_provenance.get('ingestion_method') or '-'}"),
        ("Source observed", input_provenance.get("observed_at") or "-"),
        ("Ports / points", f"{input_data.get('num_ports', '-')} / {input_data.get('frequency_count', '-')}"),
        ("Frequency range", f"{frequency_min:.3f} - {frequency_max:.3f} MHz"),
        ("Reference resistance", f"{input_data.get('reference_resistance_ohm', '-')} ohm"),
        ("Software", f"{software.get('application', '-')} API {software.get('api_version', '-')} / core {software.get('rfmatch_core_version', '-') }"),
        ("Project schema", schema),
        ("Touchstone SHA-256", input_data.get("sha256")),
        ("Project integrity", (document.get("integrity") or {}).get("digest")),
    ]
    rows: list[list[Any]] = []
    for index in range(0, len(pairs), 2):
        row: list[Any] = []
        for label, value in pairs[index:index + 2]:
            row.extend([_p(label, styles["label"]), _p(value, styles["small"])])
        while len(row) < 4:
            row.extend(["", ""])
        rows.append(row)
    return rows


def _candidate_table(candidates: list[dict], selected_index: int, styles) -> Table:
    rows: list[list[Any]] = [[
        "Candidate", "Score", "Average eta", "Minimum eta", "Component loss",
        "BOM", "Yield", "Confidence interval",
    ]]
    for index, item in enumerate(candidates):
        rows.append([
            f"{index + 1}{' selected' if index == selected_index else ''}",
            _candidate_score(item),
            _percent(item.get("avg_total_efficiency")),
            _percent(item.get("min_total_efficiency")),
            _percent(item.get("total_component_loss"), 2),
            item.get("total_component_count", "-"),
            _percent(_yield_analysis(item).get("yield_fraction")),
            _yield_interval(item),
        ])
    if not candidates:
        rows.append(["No optimization candidates saved", "", "", "", "", "", "", ""])
    return _table(
        rows,
        [24 * mm, 22 * mm, 22 * mm, 22 * mm, 22 * mm, 11 * mm, 18 * mm, 29 * mm],
        styles,
        selected_row=selected_index + 1 if candidates else None,
    )


def _manual_component_summary(components: list[dict]) -> str:
    if not components:
        return "bare DUT"
    labels = []
    for component in components:
        if component.get("part_number"):
            labels.append(str(component["part_number"]))
            continue
        kind = {
            "inductor": "L", "capacitor": "C", "resistor": "R",
            "transmission_line": "line", "open_stub": "open stub",
            "short_stub": "short stub",
        }.get(str(component.get("comp_type", "")), component.get("comp_type", "?"))
        placement = "shunt" if component.get("connection_type") == "shunt" else "series"
        value = component.get("value", component.get("electrical_length_deg", "-"))
        labels.append(f"{placement} {kind} {value}")
    return " -> ".join(labels)


def _manual_variant_table(document: dict, styles) -> Table:
    rows: list[list[Any]] = [["Name", "Port / frequency", "Network", "Return loss", "VSWR", "Input impedance", "Power error"]]
    variants = (((document.get("extensions") or {}).get("manual_workspace") or {}).get("variants") or [])
    for variant in variants:
        metrics = variant.get("metrics") or {}
        imaginary = float(metrics.get("input_impedance_imag", 0.0))
        rows.append([
            variant.get("name", "-"),
            f"P{int(variant.get('input_port', 0)) + 1} / {float(variant.get('target_frequency_hz', 0)) / 1e6:.3f} MHz",
            _manual_component_summary(variant.get("components") or []),
            _number(metrics.get("return_loss_db"), 3, " dB"),
            _number(metrics.get("vswr"), 3),
            f"{_number(metrics.get('input_impedance_real'), 2)} {'+' if imaginary >= 0 else '-'} j{abs(imaginary):.2f} ohm",
            _scientific(metrics.get("maximum_power_balance_error")),
        ])
    if len(rows) == 1:
        rows.append(["No frozen manual variants saved", "", "", "", "", "", ""])
    return _table(rows, [27 * mm, 29 * mm, 42 * mm, 19 * mm, 15 * mm, 27 * mm, 13 * mm], styles)


def _selected_metrics_table(solution: dict, styles) -> Table:
    diagnostics = _search_diagnostics(solution)
    seed = _synthesis_seed(solution)
    pairs = [
        ("Mode / objective", f"{solution.get('mode', '-')} / {solution.get('objective', '-') }"),
        ("Average total efficiency", _percent(solution.get("avg_total_efficiency"))),
        ("Minimum total efficiency", _percent(solution.get("min_total_efficiency"))),
        ("Component loss", _percent(solution.get("total_component_loss"), 2)),
        ("Power balance error", _scientific(solution.get("maximum_power_balance_error"))),
        ("Efficiency basis", solution.get("efficiency_basis")),
        ("Effective port / band priorities", _priority_weight_status(solution)),
        ("Manufacturing yield", _percent(_yield_analysis(solution).get("yield_fraction"))),
        ("Yield confidence interval", _yield_interval(solution)),
        ("P5 score margin", _number((_yield_analysis(solution).get("score_percentiles_db") or {}).get("5"), 3, " dB")),
        ("Yield variation model", _yield_variation_model(solution)),
        ("Loss-aware ideal seed", _synthesis_seed_topology(solution)),
        ("Ideal seed values", _synthesis_seed_values(solution)),
        ("Ideal seed score / evaluations", f"{_number(seed.get('score_db'), 3, ' dB')} / {seed.get('evaluations', '-') }"),
        ("Generic synthesis loss prior", _generic_synthesis_loss_status(solution)),
        ("Measured search path", _measured_search_status(solution)),
        ("Component catalog", _component_library_status(solution)),
        ("Physical evaluations / models", f"{diagnostics.get('physical_evaluations', '-')} / {diagnostics.get('component_models_loaded', '-') }"),
        ("Active physical points", diagnostics.get("active_frequency_points", "-")),
        ("Maximum components searched", diagnostics.get("maximum_components_searched", "-")),
        ("Requested search quality", (diagnostics.get("search_plan") or {}).get("label", diagnostics.get("search_quality_requested", "auto"))),
        ("Execution strategy", (diagnostics.get("search_plan") or {}).get("strategy", "hierarchical_measured")),
        ("Search budget", _number((diagnostics.get("search_plan") or {}).get("budget_seconds"), 1, " s")),
        ("Search calibration", _search_calibration_status(solution)),
        ("Calibration scope", _search_calibration_scope(solution)),
    ]
    rows = []
    for index in range(0, len(pairs), 2):
        row = []
        for label, value in pairs[index:index + 2]:
            row.extend([label, value])
        rows.append(row)
    return _table(rows, [35 * mm, 51 * mm, 35 * mm, 51 * mm], styles, header=False)


def _component_table(solution: dict, styles) -> Table:
    rows: list[list[Any]] = [["Port", "Pos.", "Connection", "Type", "Part number", "Value", "Procurement metadata"]]
    for port, metrics in _per_port(solution):
        for index, component in enumerate(metrics.get("components") or []):
            rows.append([
                f"Port {int(port) + 1}", index + 1,
                component.get("connection_type") or component.get("connection") or "-",
                component.get("type") or component.get("comp_type") or "-",
                component.get("part_number") or component.get("part") or "ideal",
                component.get("value") or "-",
                _component_procurement_metadata(component),
            ])
    if len(rows) == 1:
        rows.append(["No component data", "", "", "", "", "", ""])
    return _table(rows, [16 * mm, 14 * mm, 22 * mm, 18 * mm, 45 * mm, 20 * mm, 37 * mm], styles)


def _dependency_table(document: dict, styles) -> Table:
    rows: list[list[Any]] = [["Role", "Filename", "Location", "Transform", "SHA-256", "Bytes"]]
    for item in (document.get("configuration") or {}).get("input_dependencies") or []:
        rows.append([
            item.get("role", "-"), item.get("filename", "-"),
            item.get("location", "-"),
            ("flip; " if item.get("reverse_ports") else "") + f"{item.get('reference_impedance_mode', 'native')} Z0",
            item.get("sha256", "-"),
            item.get("size_bytes", "-"),
        ])
    if len(rows) == 1:
        rows.append(["No external layout dependencies", "", "", "", "", ""])
    return _table(rows, [18 * mm, 34 * mm, 22 * mm, 25 * mm, 58 * mm, 15 * mm], styles)


def _state_table(solution: dict, styles) -> Table:
    rows: list[list[Any]] = [["Configuration", "Active bands", "State", "Score"]]
    configurations = solution.get("frequency_configurations") or []
    for item in configurations:
        bands = " + ".join(f"{band[0]}-{band[1]} MHz" for band in item.get("bands_mhz", []))
        rows.append([item.get("name"), bands, item.get("state"), _number(item.get("score_db"), 3, " dB")])
    if not configurations:
        for name, state in (solution.get("tunable_states") or {}).items():
            rows.append([name, "-", state, "-"])
    if len(rows) == 1:
        rows.append(["Not a state-configured solution", "", "", ""])
    return _table(rows, [38 * mm, 72 * mm, 38 * mm, 24 * mm], styles)


def _yield_table(solution: dict, styles) -> Table:
    rows: list[list[Any]] = [["Configuration", "State", "Individual yield"]]
    yields = _yield_analysis(solution).get("configuration_yield_fraction") or {}
    states = solution.get("tunable_states") or {}
    for name, value in yields.items():
        rows.append([name, states.get(name, "-"), _percent(value)])
    if len(rows) == 1:
        rows.append(["Single configuration or no saved state yield", "", ""])
    return _table(rows, [62 * mm, 62 * mm, 48 * mm], styles)


def _isolation_table(solution: dict, styles) -> Table:
    rows: list[list[Any]] = [["Path", "Band", "Limit", "Worst", "Average", "Status"]]
    for target in solution.get("isolation_targets") or []:
        source = int(target.get("source_port", 0)) + 1
        destination = int(target.get("destination_port", 0)) + 1
        start = target.get("start_hz") or target.get("frequency_start_hz")
        stop = target.get("stop_hz") or target.get("frequency_stop_hz")
        band = (
            f"{float(start) / 1e9:.3f}-{float(stop) / 1e9:.3f} GHz"
            if start is not None and stop is not None else "-"
        )
        rows.append([
            f"S{destination}{source}", band,
            _number(target.get("maximum_allowed_db"), 2, " dB"),
            _number(target.get("worst_transmission_db"), 2, " dB"),
            _number(target.get("average_transmission_db"), 2, " dB"),
            "PASS" if target.get("passed") else "FAIL",
        ])
    if len(rows) == 1:
        rows.append(["No isolation targets", "", "", "", "", ""])
    return _table(rows, [24 * mm, 40 * mm, 26 * mm, 26 * mm, 26 * mm, 30 * mm], styles)


def _header_footer(canvas, document, *, title: str, digest: str, regular: str, bold: str):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(_LINE)
    canvas.setLineWidth(0.4)
    canvas.line(18 * mm, height - 15 * mm, width - 18 * mm, height - 15 * mm)
    canvas.setFont(bold, 8)
    canvas.setFillColor(_ACCENT)
    canvas.drawString(18 * mm, height - 11.5 * mm, "RF MATCHING")
    canvas.setFont(regular, 7)
    canvas.setFillColor(_MUTED)
    canvas.drawRightString(width - 18 * mm, height - 11.5 * mm, title[:72])
    canvas.line(18 * mm, 13 * mm, width - 18 * mm, 13 * mm)
    canvas.drawString(18 * mm, 8.5 * mm, f"Integrity {digest[:16]}...")
    canvas.drawRightString(width - 18 * mm, 8.5 * mm, f"Page {document.page}")
    canvas.restoreState()


def render_project_pdf(document: dict[str, Any]) -> bytes:
    """Render a validated project document as a self-contained native PDF."""
    regular, bold = _register_fonts()
    CurveChart._font_name = regular
    styles = _styles(regular, bold)
    output = BytesIO()
    title = _ascii_missing(document.get("name") or "RF Matching Project")
    digest = _ascii_missing((document.get("integrity") or {}).get("digest") or "")
    pdf = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=21 * mm,
        bottomMargin=18 * mm,
        title=title,
        author="RF Matching",
        subject="RF matching traceability report",
    )
    candidates, selected_index, selected = _selected_solution(document)
    story: list[Flowable] = [
        _p(title, styles["title"]),
        _p(
            f"Native PDF traceability report - PDF schema {PDF_REPORT_SCHEMA_VERSION} - "
            f"HTML schema {REPORT_SCHEMA_VERSION} - project updated {document.get('updated_at', '-')}",
            styles["subtitle"],
        ),
        _table(_metadata_rows(document, styles), [29 * mm, 57 * mm, 29 * mm, 57 * mm], styles, header=False),
        _section("Input dependencies", styles),
        _dependency_table(document, styles),
        _section("Candidate comparison", styles),
        _candidate_table(candidates, selected_index, styles),
        _section("Frozen manual variants", styles),
        _manual_variant_table(document, styles),
        _section("Selected solution", styles),
        _selected_metrics_table(selected, styles),
        _section("Bill of materials and topology", styles),
        _component_table(selected, styles),
        _section("Configuration and state assignment", styles),
        _state_table(selected, styles),
        _section("Configuration yield", styles),
        _yield_table(selected, styles),
        _section("Isolation targets", styles),
        _isolation_table(selected, styles),
    ]
    per_port = _per_port(selected)
    if per_port:
        story.append(_section("Stored full-physical curves", styles))
        for port, metrics in per_port:
            story.extend([
                KeepTogether([
                    CurveChart(metrics, "band_s11_db", f"Port {int(port) + 1} return loss", False),
                    Spacer(1, 2 * mm),
                    CurveChart(metrics, "band_total_eff", f"Port {int(port) + 1} total efficiency", True),
                ]),
                Spacer(1, 4 * mm),
            ])
    request = (document.get("configuration") or {}).get("tuning_request") or {}
    request_json = json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True)
    story.extend([
        PageBreak(),
        _section("Optimization configuration", styles),
        _p(request_json, styles["code"]),
        Spacer(1, 4 * mm),
        _p(
            "This self-contained report is derived from an integrity-checked project snapshot. "
            "Exact recomputation also requires the original Touchstone and component model files "
            "identified by the project metadata.",
            styles["small"],
        ),
    ])

    def page(canvas, doc):
        _header_footer(canvas, doc, title=title, digest=digest, regular=regular, bold=bold)

    pdf.build(story, onFirstPage=page, onLaterPages=page)
    return output.getvalue()
