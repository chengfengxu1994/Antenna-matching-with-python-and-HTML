"""Pydantic request models shared by API route groups."""

import math

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator
from engine.search_quality import SEARCH_QUALITY_PROFILES

from api.state import (
    DEFAULT_MURATA_DIR,
    DEFAULT_OPTENNI_COMPONENT_DIR,
    DEFAULT_SNP_DIR,
)


class DataDirConfig(BaseModel):
    snp_dir: str = DEFAULT_SNP_DIR
    murata_dir: str = DEFAULT_MURATA_DIR
    optenni_component_dir: str = DEFAULT_OPTENNI_COMPONENT_DIR
    environment_metadata_path: str = Field(default="", max_length=4096)


class SNPImportRequest(BaseModel):
    """Validated local EM/VNA Touchstone import."""

    filename: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1, max_length=64 * 1024 * 1024)
    source: Literal["CST", "HFSS", "VNA", "Touchstone", "other"] = "Touchstone"


class PortStateConfig(BaseModel):
    port_index: int
    state: str = "load"


class SinglePortConfig(BaseModel):
    port_index: int
    state: str = "load"
    use_matching: bool = False
    max_components: int = 2
    l_min_nh: float = 0.1
    l_max_nh: float = 20.0
    c_min_pf: float = 0.1
    c_max_pf: float = 20.0
    band_mhz: List[float] = Field(default_factory=lambda: [2400, 2500])
    num_band_points: int = 5


class MultiPortOptimizeRequest(BaseModel):
    snp_filename: str
    ports: List[SinglePortConfig]
    beam_width: int = 10
    timeout_seconds: float = 60.0
    optimization_goal: str = "efficiency"
    component_series: Optional[List[str]] = None


class JointOptimizeRequest(BaseModel):
    snp_filename: str
    ports: List[SinglePortConfig]
    beam_width: int = 10
    timeout_seconds: float = 120.0
    optimization_goal: str = "efficiency"


class OptimizeRequest(BaseModel):
    snp_filename: str
    target_frequency_hz: float = 64e6
    max_components: int = 4
    port_states: List[PortStateConfig] = Field(default_factory=list)
    input_port: int = 0
    topologies_filter: Optional[List[str]] = None
    beam_width: int = 10
    timeout_seconds: float = 60.0
    bands_mhz: Optional[List[List[float]]] = None
    num_band_points: int = 5


class ManualTuneRequest(BaseModel):
    snp_filename: str
    expected_network_sha256: Optional[str] = Field(default=None, min_length=64, max_length=64)
    target_frequency_hz: float = Field(default=64e6, gt=0)
    input_port: int = Field(default=0, ge=0)
    port_states: List[PortStateConfig] = Field(default_factory=list)
    components: list = Field(default_factory=list)
    sweep_start_hz: Optional[float] = None
    sweep_stop_hz: Optional[float] = None
    sweep_points: int = Field(default=100, ge=2, le=10001)
    use_snp_points: bool = True


class ManualRefineRequest(BaseModel):
    snp_filename: str
    expected_network_sha256: Optional[str] = Field(default=None, min_length=64, max_length=64)
    target_frequency_hz: float = Field(default=64e6, gt=0)
    input_port: int = Field(default=0, ge=0)
    port_states: List[PortStateConfig] = Field(default_factory=list)
    components: list = Field(default_factory=list, min_length=1, max_length=12)
    bands_mhz: List[List[float]] = Field(default_factory=lambda: [[2400, 2500]], min_length=1, max_length=8)
    target_return_loss_db: float = Field(default=10.0, ge=1.0, le=60.0)
    objective: Literal["worst", "average", "balanced"] = "balanced"
    max_passes: int = Field(default=4, ge=1, le=8)


class ManualYieldRequest(BaseModel):
    snp_filename: str
    expected_network_sha256: Optional[str] = Field(default=None, min_length=64, max_length=64)
    input_port: int = Field(default=0, ge=0)
    port_states: List[PortStateConfig] = Field(default_factory=list)
    components: list = Field(default_factory=list, max_length=12)
    bands_mhz: List[List[float]] = Field(default_factory=lambda: [[2400, 2500]], min_length=1, max_length=8)
    target_return_loss_db: float = Field(default=10.0, ge=0.0, le=60.0)
    samples: int = Field(default=200, ge=20, le=2000)
    seed: int = 1
    distribution: Literal["normal", "uniform"] = "uniform"
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    default_tolerance_pct: float = Field(default=5.0, gt=0.0, le=50.0)
    batch_correlation: float = Field(default=0.0, ge=0.0, le=1.0)


class EfficiencyLoadRequest(BaseModel):
    port_index: int = -1
    filepath: str


class EfficiencyInlineRequest(BaseModel):
    port_index: int = -1
    content: str
    filename: str = "pasted_data"


class EfficiencyClearRequest(BaseModel):
    port_index: int = -1


class ScenarioConfig(BaseModel):
    snp_filename: str
    weight: float = 1.0
    efficiency_filename: Optional[str] = None
    efficiency_kind: str = "radiation"


class MultiScenarioOptimizeRequest(BaseModel):
    scenarios: List[ScenarioConfig]
    input_port: int = 0
    bands_mhz: List[List[float]] = Field(default_factory=lambda: [[2400, 2500]])
    topology_names: List[str] = Field(default_factory=list)
    component_count: int = 2
    objective: str = "balanced"
    search_quality: Literal[
        "auto", "quick", "balanced", "thorough", "exhaustive", "custom"
    ] = "auto"
    beam_width: int = 20
    timeout_seconds: float = 120.0
    num_band_points: int = 7
    verification_band_points: int = Field(default=41, ge=2, le=161)
    max_candidates_per_position: int = 24

    @model_validator(mode="after")
    def apply_named_search_quality(self):
        profile = SEARCH_QUALITY_PROFILES.get(self.search_quality)
        if profile is not None:
            self.timeout_seconds = float(profile["timeout_seconds"])
            self.beam_width = int(profile["beam_width"])
            self.num_band_points = int(profile["num_band_points"])
        return self


class GenericSynthesisLossConfig(BaseModel):
    """Loss assumptions used only by continuous L/C topology synthesis."""

    inductor_q: float = Field(default=30.0, gt=0.0, le=1.0e6)
    inductor_q_reference_hz: float = Field(default=1.0e9, gt=0.0)
    inductor_esr_ohm: float = Field(default=0.0, ge=0.0, le=1.0e6)
    capacitor_esr_ohm: float = Field(default=0.4, ge=0.0, le=1.0e6)


class MultiScenarioManualRequest(BaseModel):
    scenarios: List[ScenarioConfig]
    input_port: int = 0
    bands_mhz: List[List[float]] = Field(default_factory=lambda: [[2400, 2500]])
    topology_name: str
    objective: str = "balanced"
    num_band_points: int = 21
    components: List[dict] = Field(default_factory=list)


class IsolationTargetConfig(BaseModel):
    """Directed S_destination,source constraint using zero-based port indices."""

    source_port: int
    destination_port: int
    band_mhz: List[float] = Field(default_factory=lambda: [2400, 2500])
    maximum_db: float = -20.0
    weight: float = 1.0
    average_weight: float = 0.0


class TunableFrequencyConfigurationConfig(BaseModel):
    """One hardware state may serve several simultaneously active bands."""

    name: str
    bands_mhz: List[List[float]]
    weight: float = 1.0


class TunableFixedComponentConfig(BaseModel):
    """Shared measured part, ordered from the DUT toward the connector."""

    connection: str = "series"
    kind: str
    value: float


class MicrostripFabricationConfig(BaseModel):
    enabled: bool = False
    substrate_name: str = Field(default="FR-4 engineering model", min_length=1, max_length=100)
    relative_permittivity: float = Field(default=4.5, gt=1.0, le=30.0)
    substrate_height_mm: float = Field(default=1.6, gt=0, le=20.0)
    loss_tangent: float = Field(default=0.02, ge=0, le=1.0)
    copper_thickness_um: float = Field(default=35.0, ge=0, le=1000.0)
    copper_resistivity_ohm_m: float = Field(default=1.68e-8, gt=0)
    copper_roughness_rms_um: float = Field(default=0.15, ge=0, le=100.0)
    minimum_width_mm: float = Field(default=0.1, gt=0, le=100.0)
    maximum_width_mm: float = Field(default=10.0, gt=0, le=1000.0)
    width_tolerance_pct: float = Field(default=10.0, ge=0.0, lt=100.0)
    length_tolerance_pct: float = Field(default=0.0, ge=0.0, lt=100.0)
    substrate_height_tolerance_pct: float = Field(default=0.0, ge=0.0, lt=100.0)
    relative_permittivity_tolerance_pct: float = Field(default=0.0, ge=0.0, lt=100.0)


class LayoutBlockConfig(BaseModel):
    filename: str = Field(min_length=1, max_length=500)
    location: Literal["dut_side", "connector_side"] = "connector_side"
    passivity_policy: Literal["warn", "reject"] = "warn"
    reverse_ports: bool = False
    reference_impedance_mode: Literal["native", "system"] = "native"
    left_fixture_filename: Optional[str] = Field(default=None, max_length=500)
    left_fixture_reverse_ports: bool = False
    right_fixture_filename: Optional[str] = Field(default=None, max_length=500)
    right_fixture_reverse_ports: bool = False
    maximum_deembedding_condition_number: float = Field(default=1e10, ge=1.0, le=1e16)


class TransmissionLineSearchConfig(BaseModel):
    characteristic_impedance_min_ohm: float = Field(default=20.0, gt=0)
    characteristic_impedance_max_ohm: float = Field(default=120.0, gt=0)
    electrical_length_min_deg: float = Field(default=1.0, gt=0, lt=180)
    electrical_length_max_deg: float = Field(default=179.0, gt=0, lt=180)
    attenuation_db: float = Field(default=0.0, ge=0)
    loss_frequency_exponent: float = Field(default=0.5, ge=0)
    topologies: List[Literal[
        "through_line", "open_stub", "short_stub",
        "connector_line_open_stub_dut", "connector_line_short_stub_dut",
        "connector_open_stub_line_dut", "connector_short_stub_line_dut",
    ]] = Field(default_factory=lambda: [
        "through_line", "open_stub", "short_stub",
        "connector_line_open_stub_dut", "connector_line_short_stub_dut",
        "connector_open_stub_line_dut", "connector_short_stub_line_dut",
    ])
    restarts: int = Field(default=10, ge=1, le=64)
    iterations: int = Field(default=24, ge=1, le=100)
    maximum_evaluations: int = Field(default=10000, ge=10, le=100000)
    microstrip: MicrostripFabricationConfig = Field(
        default_factory=MicrostripFabricationConfig
    )
    layout_blocks: List[LayoutBlockConfig] = Field(default_factory=list, max_length=8)


class ComponentParameterFilter(BaseModel):
    """Procurement constraints applied after measured-family selection."""

    manufacturers: List[str] = Field(default_factory=list, max_length=100)
    package_codes: List[str] = Field(default_factory=list, max_length=100)
    voltage_codes: List[str] = Field(default_factory=list, max_length=100)
    dielectrics: List[str] = Field(default_factory=list, max_length=100)
    maximum_tolerance_pct: Optional[float] = Field(default=None, gt=0.0, le=100.0)
    unknown_metadata_policy: Literal["include", "exclude"] = "include"


class ComponentLibraryPreviewRequest(BaseModel):
    component_series: Optional[List[str]] = None
    component_filter: ComponentParameterFilter = Field(default_factory=ComponentParameterFilter)


class ComponentAlternativesRequest(BaseModel):
    part_number: str = Field(min_length=1, max_length=200)
    component_series: Optional[List[str]] = None
    component_filter: ComponentParameterFilter = Field(default_factory=ComponentParameterFilter)
    bands_mhz: List[List[float]] = Field(default_factory=list, max_length=32)
    num_band_points: int = Field(default=5, ge=2, le=31)
    maximum_nominal_deviation_pct: float = Field(default=50.0, gt=0.0, le=1000.0)
    limit: int = Field(default=10, ge=1, le=50)


class TuningOptimizeRequest(BaseModel):
    """Unified tuning request used by the current frontend."""

    ports: List[dict] = Field(
        default_factory=lambda: [
            {
                "port_index": 0,
                "bands_mhz": [[2400, 2500]],
                "max_components": 2,
                "enabled": True,
            }
        ]
    )
    objective: str = "balanced"
    mode: str = "single"
    search_quality: Literal[
        "auto", "quick", "balanced", "thorough", "exhaustive", "custom"
    ] = "auto"
    within_band_average_weight: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Blend between the worst frequency (0) and the average in each band (1). "
            "When omitted, the selected objective preset is used."
        ),
    )
    across_band_average_weight: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Blend between the worst band (0) and the average across bands (1). "
            "When omitted, the product default of 0.1 is used."
        ),
    )
    generic_synthesis_loss: GenericSynthesisLossConfig = Field(
        default_factory=GenericSynthesisLossConfig,
        description=(
            "Generic Q/ESR assumptions for the continuous topology prior only; "
            "final product candidates use measured component S-parameters."
        ),
    )
    beam_width: int = 10
    timeout_seconds: float = 120.0
    num_band_points: int = 5
    component_series: Optional[List[str]] = None
    component_filter: ComponentParameterFilter = Field(default_factory=ComponentParameterFilter)
    band_state_map: Optional[Dict[str, List[float]]] = None
    tuner_mdif_path: Optional[str] = None
    frequency_configurations: List[TunableFrequencyConfigurationConfig] = Field(default_factory=list)
    tunable_fixed_components: List[TunableFixedComponentConfig] = Field(default_factory=list)
    tunable_auto_synthesize: bool = False
    switch_state_options: Dict[str, List[str]] = Field(default_factory=dict)
    switch_measured_refine: bool = False
    switch_max_input_components: int = Field(default=2, ge=0, le=2)
    transmission_line: TransmissionLineSearchConfig = Field(
        default_factory=TransmissionLineSearchConfig
    )
    debug: bool = False
    debug_top_n: int = 10
    isolation_targets: List[IsolationTargetConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_port_and_band_weights(self):
        normalized_ports = []
        enabled_effective_weights = []
        for index, raw_port in enumerate(self.ports):
            if not isinstance(raw_port, dict):
                raise ValueError(f"port configuration #{index + 1} must be an object")
            port = dict(raw_port)
            bands = port.get("bands_mhz") or []
            raw_weights = port.get("band_weights")
            weights = [1.0] * len(bands) if raw_weights is None else list(raw_weights)
            if len(weights) != len(bands):
                raise ValueError(
                    f"port {int(port.get('port_index', index)) + 1} band_weights must match bands_mhz"
                )
            port_weight = float(port.get("port_weight", 1.0))
            if not math.isfinite(port_weight) or not 0.0 <= port_weight <= 100.0:
                raise ValueError("port_weight must be finite and between 0 and 100")
            normalized_weights = []
            for weight in weights:
                value = float(weight)
                if not math.isfinite(value) or not 0.0 <= value <= 100.0:
                    raise ValueError("band weights must be finite and between 0 and 100")
                normalized_weights.append(value)
            port["port_weight"] = port_weight
            port["band_weights"] = normalized_weights
            normalized_ports.append(port)
            if port.get("enabled", True):
                enabled_effective_weights.extend(port_weight * value for value in normalized_weights)
        if enabled_effective_weights and not any(value > 0.0 for value in enabled_effective_weights):
            raise ValueError("at least one enabled optimization band must have a positive effective weight")
        self.ports = normalized_ports
        return self

    @model_validator(mode="after")
    def apply_named_search_quality(self):
        """Named product profiles are authoritative; custom/auto keep raw knobs."""
        profile = SEARCH_QUALITY_PROFILES.get(self.search_quality)
        if profile is not None:
            self.timeout_seconds = float(profile["timeout_seconds"])
            self.beam_width = int(profile["beam_width"])
            self.num_band_points = int(profile["num_band_points"])
        return self


class TuningYieldRequest(BaseModel):
    """Monte Carlo tolerance analysis for the current measured candidates."""

    solution_indices: Optional[List[int]] = None
    samples: int = Field(default=200, ge=20, le=5000)
    seed: int = 1
    distribution: Literal["normal", "uniform"] = "uniform"
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    minimum_total_efficiency: float = Field(default=0.5, ge=0.0, le=1.0)
    minimum_average_total_efficiency: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Geometric mean efficiency threshold (arithmetic average in dB)",
    )
    minimum_return_loss_db: float = Field(default=6.0, ge=0.0)
    default_tolerance_pct: float = Field(default=5.0, gt=0.0, le=50.0)
    batch_correlation: float = Field(default=0.0, ge=0.0, le=1.0)
    reference_temperature_c: float = Field(default=25.0, ge=-273.15, le=500.0)
    temperature_min_c: Optional[float] = Field(default=None, ge=-273.15, le=500.0)
    temperature_max_c: Optional[float] = Field(default=None, ge=-273.15, le=500.0)
    inductor_tempco_ppm_per_c: float = Field(default=0.0, ge=-10000.0, le=10000.0)
    capacitor_tempco_ppm_per_c: float = Field(default=0.0, ge=-10000.0, le=10000.0)
    inductor_bias_pct: float = Field(default=0.0, gt=-100.0, le=1000.0)
    capacitor_bias_pct: float = Field(default=0.0, gt=-100.0, le=1000.0)


class TuneSingleRequest(BaseModel):
    snp_filename: str = ""
    port_index: int = 0
    bands_mhz: List[List[float]] = Field(default_factory=lambda: [[2400, 2500]])
    band_weights: Optional[List[float]] = None
    port_weight: float = Field(default=1.0, ge=0.0, le=100.0)
    max_components: int = 2
    objective: str = "average_efficiency"
    within_band_average_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    across_band_average_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    generic_synthesis_loss: GenericSynthesisLossConfig = Field(
        default_factory=GenericSynthesisLossConfig
    )
    topology_filter: Optional[List[str]] = None
    component_series: Optional[List[str]] = None
    beam_width: int = 20
    timeout_seconds: float = 60.0
    num_band_points: int = 10

    @model_validator(mode="after")
    def validate_band_weights(self):
        if self.band_weights is None:
            self.band_weights = [1.0] * len(self.bands_mhz)
        if len(self.band_weights) != len(self.bands_mhz):
            raise ValueError("band_weights must match bands_mhz")
        if any(not math.isfinite(value) or not 0.0 <= value <= 100.0 for value in self.band_weights):
            raise ValueError("band weights must be finite and between 0 and 100")
        if self.bands_mhz and not any(self.port_weight * value > 0.0 for value in self.band_weights):
            raise ValueError("at least one optimization band must have a positive effective weight")
        return self


class TuneJointRequest(BaseModel):
    ports: List[dict]
    objective: str = "balanced"
    beam_width: int = 10
    timeout_seconds: float = 120.0
    num_band_points: int = 5


class TunePowerBalanceRequest(BaseModel):
    port_configs: Dict[int, List[dict]] = Field(default_factory=dict)
    frequency_hz: float = 2.45e9


class ManualVariantMetrics(BaseModel):
    return_loss_db: float = Field(ge=0.0)
    return_loss_improvement_db: float
    vswr: float = Field(ge=1.0)
    input_impedance_real: float
    input_impedance_imag: float
    maximum_power_balance_error: float = Field(ge=0.0)
    numeric_core: str = Field(min_length=1, max_length=100)


class ManualVariantSnapshot(BaseModel):
    variant_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    name: str = Field(min_length=1, max_length=100)
    input_port: int = Field(ge=0, le=63)
    target_frequency_hz: float = Field(gt=0.0)
    components: List[Dict[str, object]] = Field(default_factory=list, max_length=12)
    port_states: List[PortStateConfig] = Field(default_factory=list, max_length=64)
    metrics: ManualVariantMetrics
    created_at: str = Field(min_length=1, max_length=64)


class ManualWorkspaceSnapshot(BaseModel):
    schema_version: Literal[1] = 1
    active_input_port: int = Field(default=0, ge=0, le=63)
    target_frequency_hz: float = Field(gt=0.0)
    working_networks: Dict[str, List[Dict[str, object]]] = Field(
        default_factory=dict, max_length=64,
    )
    variants: List[ManualVariantSnapshot] = Field(default_factory=list, max_length=12)
    selected_variant_id: Optional[str] = Field(default=None, max_length=64)
    overlay_variant_ids: List[str] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def validate_workspace_references(self):
        for port, components in self.working_networks.items():
            if not port.isdigit() or not 0 <= int(port) <= 63:
                raise ValueError("manual working network port key is invalid")
            if len(components) > 12:
                raise ValueError("manual working network supports at most 12 components per port")
        variant_ids = [variant.variant_id for variant in self.variants]
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("manual variant_id values must be unique")
        if self.selected_variant_id is not None and self.selected_variant_id not in variant_ids:
            raise ValueError("selected manual variant does not exist")
        if len(self.overlay_variant_ids) != len(set(self.overlay_variant_ids)):
            raise ValueError("manual overlay variant IDs must be unique")
        if any(variant_id not in variant_ids for variant_id in self.overlay_variant_ids):
            raise ValueError("manual overlay variant does not exist")
        return self


class ProjectSaveRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    project_id: Optional[str] = Field(default=None, max_length=64)
    manual_workspace: Optional[ManualWorkspaceSnapshot] = None


class ProjectLoadRequest(BaseModel):
    project_id: str = Field(min_length=1, max_length=64)
    verify_input: bool = True


class ProjectImportRequest(BaseModel):
    document: Dict[str, object]
    conflict_policy: Literal["copy", "replace", "reject"] = "copy"


class ProjectRelinkRequest(BaseModel):
    project_id: str = Field(min_length=1, max_length=64)
    apply_matches: bool = True


class TuningContinueRequest(BaseModel):
    additional_seconds: float = Field(default=30.0, gt=0.0, le=3600.0)
