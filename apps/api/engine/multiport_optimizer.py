"""
Joint Multi-Port Matching Optimizer.

Unlike the single-port optimizer that treats each port independently,
this optimizer searches the COMBINED space of matching components for
ALL ports simultaneously, properly accounting for S21/S12 coupling.

Algorithm:
  Phase 1: Independent candidate generation per port (reuse MatchingOptimizer)
  Phase 2: Joint evaluation — apply ALL ports' matching to full S-matrix
  Phase 3: Optional local refinement

System metrics (per port i):
  η_mismatch_i = 1 - |S'_ii|²      (accepted power efficiency)
  η_total_i    = η_rad_i × η_mismatch_i  (if radiation efficiency data)
  η_isolation  = Σ(j≠i) |S'_ji|²    (power lost to coupling)
  η_system_i   = η_total_i - η_isolation   (true radiated efficiency)

Where S' is the full S-matrix AFTER all matching networks are applied.
"""

import numpy as np
import logging
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import time
import itertools
from project_paths import ARTIFACTS_DIR

from .touchstone import TouchstoneData
from .network import (
    s_to_z, s_to_y, y_to_s,
    _embed_series_on_port, _embed_shunt_to_ground, connect_2port_to_multiport
)
from .component_lib import ComponentLibrary, ComponentInfo
from .topology import Topology, TopologyElement, ConnectionType, get_standard_topologies
from .optimizer import (
    MatchingOptimizer, OptimizerConfig, MatchingSolution,
    PortState, ComponentChoice, TERMINATION_GAMMA, _solution_rank_key
)
from .efficiency_data import EfficiencyData
from .network import terminate_ports
from .cost_function import (
    get_optimization_mode, OptimizationMode,
    estimate_total_component_loss, compute_unified_score, ScoreInput,
)
from rfmatch_core import PORT_TOPOLOGY_PATTERNS

logger = logging.getLogger("rf_matching.optimizer")


def optenni_compatible_topologies(max_components: int) -> List[Topology]:
    """Return the bounded 1–4 element synthesis set shared with rfmatch-core."""
    allowed_patterns = {
        tuple((connection, "inductor" if kind == "L" else "capacitor") for connection, kind in pattern)
        for pattern in PORT_TOPOLOGY_PATTERNS
        if pattern and len(pattern) <= max_components
    }
    return [
        topology for topology in get_standard_topologies()
        if topology.num_components <= max_components
        and tuple(
            (element.connection_type.value, element.component_type)
            for element in topology.elements
        ) in allowed_patterns
    ]

# Ensure the optimizer.log file handler exists for structured logging
_log_dir = str(ARTIFACTS_DIR / 'output')
os.makedirs(_log_dir, exist_ok=True)
_log_path = os.path.join(_log_dir, 'optimizer.log')
if not any(isinstance(h, logging.FileHandler) and h.baseFilename == _log_path
           for h in logger.handlers):
    _fh = logging.FileHandler(_log_path, mode='a', encoding='utf-8')
    _fh.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s'))
    _fh.setLevel(logging.DEBUG)
    logger.addHandler(_fh)


def _component_label(choice: ComponentChoice) -> str:
    comp = choice.component
    value = getattr(comp, "nominal_value", None)
    unit = getattr(comp, "nominal_unit", "")
    part = getattr(comp, "part_number", "")
    if value is None:
        value_text = "?"
    else:
        value_text = f"{value:g}{unit}"
    return f"{choice.connection_type}:{value_text}:{part}"


def _solution_label(solution: MatchingSolution) -> str:
    comps = ", ".join(_component_label(c) for c in solution.component_choices)
    return f"{solution.topology.name} [{comps}]"


@dataclass
class PortMatchConfig:
    """Configuration for one port's matching network."""
    port_index: int
    max_components: int = 2
    target_frequency_hz: float = 64e6
    # Topology filter (None = all standard topologies)
    topology_names: Optional[List[str]] = None


@dataclass
class JointSolution:
    """A complete multi-port matching solution."""
    # Per-port matching solutions
    port_solutions: Dict[int, MatchingSolution]  # port_index → solution
    # Full system S-matrix after all matching applied
    system_s_matrix: Optional[np.ndarray] = None
    # Per-port system metrics (after ALL matching applied)
    port_metrics: Dict[int, dict] = field(default_factory=dict)
    # System-level metrics
    min_system_efficiency: float = 0.0   # Worst port's mismatch efficiency
    avg_system_efficiency: float = 0.0   # Average mismatch efficiency across ports
    min_total_efficiency: float = 0.0    # Worst port's total efficiency (with η_rad)
    avg_total_efficiency: float = 0.0    # Average total efficiency
    max_coupling_loss: float = 0.0       # Max |S'_ji|² across all port pairs
    balanced_score: float = 0.0          # Final ranking score
    # Power balance breakdown
    component_loss_total: float = 0.0    # Total estimated component dissipation
    # Per-port power balance (reflected, coupled, component_loss, radiated)
    power_balance: Dict[int, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'port_solutions': {
                str(k): v.to_dict() for k, v in self.port_solutions.items()
            },
            'port_metrics': {
                str(k): v for k, v in self.port_metrics.items()
            },
            'min_system_efficiency': self.min_system_efficiency,
            'avg_system_efficiency': self.avg_system_efficiency,
            'min_total_efficiency': self.min_total_efficiency,
            'avg_total_efficiency': self.avg_total_efficiency,
            'max_coupling_loss': self.max_coupling_loss,
            'balanced_score': self.balanced_score,
            'component_loss_total': self.component_loss_total,
            'power_balance': {
                str(k): v for k, v in self.power_balance.items()
            },
            # Convenience: component summary
            'components_summary': {
                str(k): [
                    {
                        'part': c.component.part_number,
                        'type': c.connection_type,
                        'value': f"{c.component.nominal_value}{c.component.nominal_unit}",
                    }
                    for c in v.component_choices
                ]
                for k, v in self.port_solutions.items()
            },
        }


def evaluate_joint_solution(
    dut: TouchstoneData,
    port_configs: Dict[int, List[ComponentChoice]],
    freq_hz: float,
    Z0: float = 50.0,
    radiation_efficiency: Optional[EfficiencyData] = None,
    per_port_efficiency: Optional[Dict[int, EfficiencyData]] = None,
) -> dict:
    """
    Evaluate a joint multi-port matching solution.
    
    Applies ALL matching networks to the full N-port S-matrix sequentially,
    then computes system efficiency for each port.
    
    Args:
        dut: Raw antenna S-parameter data
        port_configs: {port_index: [ComponentChoice, ...]} for each matched port
        freq_hz: Evaluation frequency
        Z0: Reference impedance
        radiation_efficiency: Optional η_rad data (applied to all ports)
        per_port_efficiency: Optional per-port η_rad data {port_index: EfficiencyData}
    
    Returns:
        Dict with per-port metrics and system-level metrics
    """
    # Get the full S-matrix at the evaluation frequency
    S = dut.get_s_matrix_interpolated(freq_hz)
    n_ports = S.shape[0]
    
    # Track component S-params for loss estimation
    all_component_s_params = []  # List of (S_matrix, connection_type)
    
    # Apply ALL matching networks sequentially
    for port_idx, choices in port_configs.items():
        for choice in choices:
            comp_s = choice.component.get_s_matrix_at_freq(freq_hz)
            all_component_s_params.append((comp_s, choice.connection_type))
            try:
                if choice.connection_type == 'series':
                    S = _embed_series_on_port(S, comp_s, choice.port)
                elif choice.connection_type == 'shunt':
                    S = _embed_shunt_to_ground(S, comp_s, choice.port)
                elif choice.connection_type == 'parallel':
                    S = connect_2port_to_multiport(S, comp_s, choice.port, choice.port2)
            except (np.linalg.LinAlgError, Exception):
                return {'valid': False, 'reason': 'Singular matrix during embedding'}
    
    # Compute per-port metrics from the FULL system S-matrix
    port_metrics = {}
    mismatch_effs = []
    total_effs = []
    coupling_losses = []
    power_balance = {}
    
    # Compute total component loss estimate BEFORE per-port loop
    # so we can include it in per-port total_efficiency
    total_comp_loss = estimate_total_component_loss(all_component_s_params)
    n_matched = len(port_configs)
    
    for i in range(n_ports):
        sii = S[i, i]
        mismatch_eff = 1.0 - abs(sii) ** 2
        
        # Coupling loss: power lost to other ports
        coupling_loss = sum(abs(S[j, i]) ** 2 for j in range(n_ports) if j != i)
        
        # Radiated power = accepted - coupled
        radiated_eff = max(0.0, mismatch_eff - coupling_loss)
        
        # Per-port component loss allocation
        comp_loss_per_port = total_comp_loss / max(n_matched, 1) if i in port_configs else 0.0
        
        # Total efficiency = radiated - component loss (consistent with power balance)
        total_eff = max(0.0, radiated_eff - comp_loss_per_port)
        
        port_info = {
            's11': complex(sii),
            's11_db': float(-20 * np.log10(max(abs(sii), 1e-15))),
            's11_magnitude': float(abs(sii)),
            'mismatch_efficiency': float(mismatch_eff),
            'coupling_loss': float(coupling_loss),
            'radiated_efficiency': float(radiated_eff),
            'total_efficiency': float(total_eff),
            'coupling_to_ports': {
                str(j): float(abs(S[j, i]) ** 2)
                for j in range(n_ports) if j != i
            },
        }
        
        port_metrics[i] = port_info
        mismatch_effs.append(mismatch_eff)
        total_effs.append(total_eff)
        coupling_losses.append(coupling_loss)
    
    # Power balance breakdown per port (uses eta_rad for antenna loss)
    for i in range(n_ports):
        pm = port_metrics.get(i, {})
        ref_power = 1.0 - pm.get('mismatch_efficiency', 1.0)  # reflected
        coup_power = pm.get('coupling_loss', 0.0)  # coupled to other ports
        # Allocate component loss proportionally
        comp_power = total_comp_loss / max(n_matched, 1) if i in port_configs else 0.0
        rad_power = max(0.0, pm.get('radiated_efficiency', 0.0) - comp_power)
        # Antenna radiation efficiency (if available)
        if per_port_efficiency and i in per_port_efficiency:
            eta_rad_pb = per_port_efficiency[i].get_efficiency_at(freq_hz)
            ant_loss = rad_power * (1.0 - eta_rad_pb)
            rad_power = rad_power * eta_rad_pb
        elif radiation_efficiency:
            eta_rad_pb = radiation_efficiency.get_efficiency_at(freq_hz)
            ant_loss = rad_power * (1.0 - eta_rad_pb)
            rad_power = rad_power * eta_rad_pb
        else:
            ant_loss = 0.0
        
        power_balance[i] = {
            'reflected': float(ref_power),
            'coupled': float(coup_power),
            'component_loss': float(comp_power),
            'antenna_loss': float(ant_loss),
            'radiated': float(rad_power),
            # Normalise to sum = 1.0
            'total': float(ref_power + coup_power + comp_power + ant_loss + rad_power),
        }
    
    return {
        'valid': True,
        'port_metrics': port_metrics,
        's_matrix': S,
        'min_mismatch_efficiency': float(np.min(mismatch_effs)),
        'avg_mismatch_efficiency': float(np.mean(mismatch_effs)),
        'min_total_efficiency': float(np.min(total_effs)),
        'avg_total_efficiency': float(np.mean(total_effs)),
        'max_coupling_loss': float(np.max(coupling_losses)),
        'component_loss_total': float(total_comp_loss),
        'power_balance': power_balance,
        'component_s_params': [
            {'s_matrix': s.tolist(), 'connection_type': ct}
            for s, ct in all_component_s_params
        ],
    }


class JointMultiPortOptimizer:
    """
    Joint multi-port matching optimizer.
    
    Searches the combined space of matching components for all ports
    simultaneously, accounting for cross-port coupling (S21/S12).
    """
    
    def __init__(
        self,
        dut: TouchstoneData,
        component_library: ComponentLibrary,
        port_configs: List[PortMatchConfig],
        Z0: float = 50.0,
        radiation_efficiency: Optional[EfficiencyData] = None,
        per_port_efficiency: Optional[Dict[int, EfficiencyData]] = None,
        min_avg_balance: float = 0.5,
        top_candidates_per_port: int = 8,
        timeout_seconds: float = 120.0,
        optimization_mode: str = 'efficiency',
        debug: bool = False,
        debug_top_n: int = 10,
    ):
        self.dut = dut
        self.library = component_library
        self.port_configs = {pc.port_index: pc for pc in port_configs}
        self.Z0 = Z0
        self.radiation_efficiency = radiation_efficiency
        self.per_port_efficiency = per_port_efficiency or {}
        self.min_avg_balance = min_avg_balance
        self.top_K = top_candidates_per_port
        self.timeout_seconds = timeout_seconds
        self.optimization_mode = get_optimization_mode(optimization_mode)
        # Debug support
        self.debug = debug
        self.debug_top_n = debug_top_n
        self._debug_info = {}
    
    def _phase1_independent_candidates(self) -> Dict[int, List[MatchingSolution]]:
        """
        Phase 1: Generate top-K independent matching candidates per port.
        Uses the existing single-port optimizer with topology-bucketed diversity.
        """
        candidates = {}
        
        for port_idx, pc in self.port_configs.items():
            logger.info(
                "╔══════════════════════════════════════════════════════════╗")
            logger.info(
                "║  PHASE1  port=%d  target=%.3f MHz  max_comp=%d  top_K=%d  ║",
                port_idx,
                pc.target_frequency_hz / 1e6,
                pc.max_components,
                self.top_K,
            )
            logger.info(
                "╚══════════════════════════════════════════════════════════╝")
            config = OptimizerConfig(
                target_frequency_hz=pc.target_frequency_hz,
                max_components=pc.max_components,
                reference_impedance=self.Z0,
                beam_width=self.top_K,
                timeout_seconds=self.timeout_seconds / len(self.port_configs),
                # Don't use bands for individual candidate generation
                bands_mhz=None,
                max_combinations_to_evaluate=5000,
                progressive_candidates_per_type=32,
                fine_candidates_per_position=8,
            )
            
            opt = MatchingOptimizer(self.dut, self.library, config)
            
            # Build port states: terminate all non-input ports with 50Ω load
            port_states = {}
            n_ports = self.dut.num_ports
            for other_idx in range(n_ports):
                if other_idx != port_idx:
                    port_states[other_idx] = PortState.LOAD
            
            # Get topologies
            topos = optenni_compatible_topologies(pc.max_components)
            if pc.topology_names:
                topos = [t for t in topos if t.name in pc.topology_names]
            
            # ── Run optimizer per topology, collect per-topology buckets ──
            logger.info("PHASE1 port=%d: running %d topologies", port_idx, len(topos))
            topo_buckets = {}  # topology_name -> [MatchingSolution, ...]
            port_deadline = time.time() + self.timeout_seconds / max(len(self.port_configs), 1)
            for topo in topos:
                if time.time() >= port_deadline:
                    logger.warning("PHASE1 port=%d reached its time budget", port_idx)
                    break
                sols = opt.optimize_with_topology(topo, port_states, input_port=port_idx)
                if sols:
                    topo_buckets[topo.name] = sols
                    # Task 2: per-topology top-5 diagnostic
                    for rank, sol in enumerate(sols[: min(5, len(sols))], start=1):
                        accepted_eff = 1.0 - sol.s11_magnitude ** 2
                        logger.debug(
                            "  DBG port=%d topology=%s #%d RL=%.2fdB |Sii|=%.5f "
                            "accepted=%.4f comps=%s",
                            port_idx, topo.name, rank, sol.s11_db, sol.s11_magnitude,
                            accepted_eff, _solution_label(sol),
                        )
                    logger.info(
                        "  topology=%s: %d solutions, best RL=%.2fdB %s",
                        topo.name, len(sols),
                        sols[0].s11_db, _solution_label(sols[0]),
                    )
                else:
                    logger.info("  topology=%s: 0 solutions", topo.name)
            
            # ── Diversity-bucketed candidate selection (Task 4) ──
            # At least M from each topology, then fill remaining from best overall
            top_M = max(1, self.top_K // max(len(topo_buckets), 1))
            selected = []
            seen_sigs = set()  # dedup: (topology, (conn, value) for each comp)
            
            def _solution_signature(sol):
                """Signature for dedup: topology name + (conn_type, value) sorted."""
                items = tuple(sorted(
                    (c.connection_type, round(c.component.nominal_value, 2))
                    for c in sol.component_choices
                ))
                return (sol.topology.name, items)
            
            # First pass: take top_M from each topology bucket (strict limit)
            for topo_name in sorted(topo_buckets.keys()):
                bucket = topo_buckets[topo_name]
                taken = 0
                for sol in bucket:
                    if taken >= top_M:
                        break
                    sig = _solution_signature(sol)
                    if sig not in seen_sigs:
                        seen_sigs.add(sig)
                        selected.append(sol)
                        taken += 1
                        logger.debug(
                            "  DIVERSITY port=%d select %s (bucket: %s, taken=%d/%d)",
                            port_idx, _solution_label(sol), topo_name, taken, top_M,
                        )
            
            # Second pass: fill remaining from best across all topologies
            if len(selected) < self.top_K:
                all_sols = []
                for bucket in topo_buckets.values():
                    all_sols.extend(bucket)
                all_sols.sort(key=lambda s: _solution_rank_key(s, self.library))
                for sol in all_sols:
                    sig = _solution_signature(sol)
                    if sig not in seen_sigs and len(selected) < self.top_K:
                        seen_sigs.add(sig)
                        selected.append(sol)
                        logger.debug(
                            "  DIVERSITY port=%d fill %s",
                            port_idx, _solution_label(sol),
                        )
            
            # Rank single-port candidates with a mild endpoint-value penalty.
            selected.sort(key=lambda s: _solution_rank_key(s, self.library))
            logger.info(
                "PHASE1 port=%d: %d diverse candidates selected (from %d topology buckets)",
                port_idx, len(selected), len(topo_buckets),
            )

            # Remap component ports from 0 (local reduced-matrix index) to port_idx
            # (original DUT port), so joint evaluation on the full N-port S-matrix
            # embeds components on the correct port.
            remapped = []
            for sol in selected:
                new_choices = []
                for c in sol.component_choices:
                    new_choices.append(ComponentChoice(
                        position=c.position,
                        component=c.component,
                        connection_type=c.connection_type,
                        port=port_idx if c.port == 0 else c.port,
                        port2=port_idx if c.port2 is not None and c.port2 == 0 else c.port2,
                    ))
                remapped.append(MatchingSolution(
                    topology=sol.topology,
                    port_states=sol.port_states,
                    component_choices=new_choices,
                    s11_magnitude=sol.s11_magnitude,
                    s11_complex=sol.s11_complex,
                    frequency_hz=sol.frequency_hz,
                    input_impedance=sol.input_impedance,
                    vswr=sol.vswr,
                    return_loss_db=sol.return_loss_db,
                    band_efficiency=sol.band_efficiency,
                    avg_band_efficiency=sol.avg_band_efficiency,
                    min_band_efficiency=sol.min_band_efficiency,
                    balanced_score=sol.balanced_score,
                    avg_total_efficiency=sol.avg_total_efficiency,
                    min_total_efficiency=sol.min_total_efficiency,
                ))
            
            final_count = min(len(remapped), self.top_K)
            candidates[port_idx] = remapped[:final_count]
            
            # Structured phase1 summary
            logger.info(
                "╔══ PHASE1 port=%d final candidates (%d) ══╗",
                port_idx, final_count,
            )
            for rank, sol in enumerate(candidates[port_idx], start=1):
                accepted_eff = 1.0 - sol.s11_magnitude ** 2
                logger.info(
                    "  cand#%02d  RL=%.2fdB  |Sii|=%.5f  accepted=%.4f  %s",
                    rank, sol.s11_db, sol.s11_magnitude, accepted_eff,
                    _solution_label(sol),
                )
            logger.info(
                "╚═══════════════════════════════════════════════╝")
        
        return candidates
    
    def _phase2_joint_evaluation(
        self,
        candidates: Dict[int, List[MatchingSolution]],
    ) -> List[JointSolution]:
        """
        Phase 2: jointly evaluate per-port candidates.

        Full Cartesian evaluation is used while the product is practical. For
        very large products, use a deterministic beam seeded by independent
        match quality, then run coordinate refinement with the same full
        multi-port S-matrix evaluator.
        """
        port_indices = sorted(candidates.keys())
        candidate_lists = [candidates[pi] for pi in port_indices]
        total_combos = int(np.prod([len(c) for c in candidate_lists])) if candidate_lists else 0
        full_limit = max(1, min(200000, self.top_K ** max(2, len(port_indices)) * 64))
        logger.info(
            "PHASE2 ports=%s counts=%s total_combos=%d full_limit=%d",
            port_indices,
            [len(c) for c in candidate_lists],
            total_combos,
            full_limit,
        )

        evaluated = 0
        joint_solutions: List[JointSolution] = []

        if total_combos <= full_limit:
            for combo in itertools.product(*candidate_lists):
                evaluated += 1
                js = self._evaluate_joint_combo(port_indices, combo)
                if js is not None:
                    joint_solutions.append(js)
            search_mode = "full"
        else:
            search_mode = "beam"
            joint_solutions, evaluated = self._phase2_beam_evaluation(
                port_indices,
                candidate_lists,
                full_limit,
            )

        refined, refine_evals = self._refine_joint_solutions(
            port_indices,
            candidate_lists,
            joint_solutions[: max(4, min(12, self.top_K))],
        )
        evaluated += refine_evals
        joint_solutions.extend(refined)
        joint_solutions = self._dedup_joint_solutions(joint_solutions)
        joint_solutions.sort(key=lambda s: s.balanced_score, reverse=True)
        self._log_phase2_results(port_indices, joint_solutions, evaluated, search_mode)
        return joint_solutions

    def _evaluate_joint_combo(
        self,
        port_indices: List[int],
        combo: Tuple[MatchingSolution, ...],
    ) -> Optional[JointSolution]:
        """Evaluate one concrete multi-port combination with the full S-matrix."""
        port_comp_choices = {}
        port_solutions = {}
        for pi, sol in zip(port_indices, combo):
            port_comp_choices[pi] = sol.component_choices
            port_solutions[pi] = sol

        eval_freq = self.port_configs[port_indices[0]].target_frequency_hz
        result = evaluate_joint_solution(
            self.dut,
            port_comp_choices,
            eval_freq,
            self.Z0,
            self.radiation_efficiency,
            self.per_port_efficiency,
        )
        if not result.get('valid', False):
            return None

        n_ports = self.dut.num_ports
        port_metrics_dict = result['port_metrics']
        avg_port_eff = []
        min_port_eff = []
        for i in range(n_ports):
            pm = port_metrics_dict.get(i, {})
            eff = pm.get('total_efficiency', pm.get('mismatch_efficiency', 0))
            avg_port_eff.append(eff)
            min_port_eff.append(eff)

        all_effs = [
            pm.get('total_efficiency', pm.get('mismatch_efficiency', 0))
            for pm in port_metrics_dict.values()
        ]

        comp_s_params = []
        for sol in combo:
            for c in sol.component_choices:
                try:
                    cs = c.component.get_s_matrix_at_freq(eval_freq)
                    comp_s_params.append((cs, c.connection_type))
                except Exception:
                    pass
        comp_loss = estimate_total_component_loss(comp_s_params)

        score_input = ScoreInput(
            avg_port_efficiency=np.array(avg_port_eff),
            min_port_efficiency=np.array(min_port_eff),
            avg_band_efficiency=float(np.mean(all_effs)) if all_effs else 0.0,
            min_band_efficiency=float(np.min(all_effs)) if all_effs else 0.0,
            max_coupling_loss=result['max_coupling_loss'],
            component_loss=comp_loss,
            component_count=sum(len(sol.component_choices) for sol in combo),
        )

        return JointSolution(
            port_solutions=port_solutions,
            system_s_matrix=result.get('s_matrix'),
            port_metrics=result['port_metrics'],
            min_system_efficiency=result['min_mismatch_efficiency'],
            avg_system_efficiency=result['avg_mismatch_efficiency'],
            min_total_efficiency=result['min_total_efficiency'],
            avg_total_efficiency=result['avg_total_efficiency'],
            max_coupling_loss=result['max_coupling_loss'],
            balanced_score=compute_unified_score(score_input, self.optimization_mode),
            component_loss_total=result.get('component_loss_total', 0.0),
            power_balance=result.get('power_balance', {}),
        )

    def _phase2_beam_evaluation(
        self,
        port_indices: List[int],
        candidate_lists: List[List[MatchingSolution]],
        max_evaluations: int,
    ) -> Tuple[List[JointSolution], int]:
        """
        Guided joint search for very large products.

        Partial combinations are retained by independent per-port quality, then
        complete candidates are ranked by the real joint S-matrix evaluation.
        """
        beam_width = max(32, self.top_K * 12)
        beam: List[Tuple[MatchingSolution, ...]] = [tuple()]

        for depth, cand_list in enumerate(candidate_lists):
            expanded = [partial + (cand,) for partial in beam for cand in cand_list]
            if depth < len(candidate_lists) - 1 and len(expanded) > beam_width:
                expanded.sort(key=self._independent_combo_key)
                expanded = expanded[:beam_width]
            beam = expanded

        complete = beam
        if len(complete) > max_evaluations:
            complete.sort(key=self._independent_combo_key)
            complete = complete[:max_evaluations]

        joint_solutions = []
        evaluated = 0
        for combo in complete:
            evaluated += 1
            js = self._evaluate_joint_combo(port_indices, combo)
            if js is not None:
                joint_solutions.append(js)
        return joint_solutions, evaluated

    def _independent_combo_key(self, combo: Tuple[MatchingSolution, ...]) -> Tuple[float, int]:
        return (
            sum(_solution_rank_key(sol, self.library) for sol in combo),
            sum(len(sol.component_choices) for sol in combo),
        )

    def _refine_joint_solutions(
        self,
        port_indices: List[int],
        candidate_lists: List[List[MatchingSolution]],
        seeds: List[JointSolution],
        max_passes: int = 3,
    ) -> Tuple[List[JointSolution], int]:
        """Coordinate refinement: replace one port at a time and keep improvements."""
        refined: List[JointSolution] = []
        evaluated = 0

        for seed in seeds:
            combo = tuple(seed.port_solutions[pi] for pi in port_indices)
            current = seed
            for _ in range(max_passes):
                changed = False
                for pos, cand_list in enumerate(candidate_lists):
                    best = current
                    best_combo = combo
                    for cand in cand_list:
                        if cand is combo[pos]:
                            continue
                        trial = list(combo)
                        trial[pos] = cand
                        evaluated += 1
                        js = self._evaluate_joint_combo(port_indices, tuple(trial))
                        if js is not None and js.balanced_score > best.balanced_score + 1e-12:
                            best = js
                            best_combo = tuple(trial)
                    if best is not current:
                        current = best
                        combo = best_combo
                        changed = True
                if not changed:
                    break
            refined.append(current)

        return refined, evaluated

    def _dedup_joint_solutions(self, solutions: List[JointSolution]) -> List[JointSolution]:
        seen = set()
        unique = []
        for sol in sorted(solutions, key=lambda s: s.balanced_score, reverse=True):
            sig = tuple(
                (
                    pi,
                    tuple(
                        (
                            c.position,
                            c.connection_type,
                            c.component.part_number,
                            getattr(c.component, "s2p_filename", ""),
                            getattr(c.component, "zip_path", ""),
                        )
                        for c in sol.port_solutions[pi].component_choices
                    ),
                )
                for pi in sorted(sol.port_solutions)
            )
            if sig in seen:
                continue
            seen.add(sig)
            unique.append(sol)
        return unique

    def _log_phase2_results(
        self,
        port_indices: List[int],
        joint_solutions: List[JointSolution],
        evaluated: int,
        search_mode: str,
    ) -> None:
        logger.info(
            "PHASE2 mode=%s evaluated=%d valid=%d top=%d",
            search_mode,
            evaluated,
            len(joint_solutions),
            min(10, len(joint_solutions)),
        )
        for rank, sol in enumerate(joint_solutions[: min(10, len(joint_solutions))], start=1):
            comp_summary = []
            for pi in port_indices:
                port_sol = sol.port_solutions.get(pi)
                if port_sol:
                    comp_summary.append(f"P{pi + 1}:{_solution_label(port_sol)}")
            enabled_eff = {
                pi: sol.port_metrics.get(pi, {}).get(
                    "total_efficiency",
                    sol.port_metrics.get(pi, {}).get("mismatch_efficiency", 0.0),
                )
                for pi in port_indices
            }
            logger.info(
                "  #%02d score=%.5f avg_eff=%.5f min_eff=%.5f coupling=%.5f comp_loss=%.5f eff=%s %s",
                rank,
                sol.balanced_score,
                sol.avg_total_efficiency,
                sol.min_total_efficiency,
                sol.max_coupling_loss,
                sol.component_loss_total,
                {k: round(v, 5) for k, v in enabled_eff.items()},
                " | ".join(comp_summary),
            )

    def _evaluate_at_band_frequencies(
        self,
        joint_sol: JointSolution,
        bands_mhz: List[List[float]],
        num_points: int = 5,
    ) -> JointSolution:
        """
        Re-evaluate a joint solution across frequency bands.
        Updates the port_metrics and efficiency fields with band-averaged values.
        """
        all_mismatch = []
        all_total = []
        
        for band in bands_mhz:
            freqs = np.linspace(band[0] * 1e6, band[1] * 1e6, num_points)
            
            for freq_hz in freqs:
                port_comp_choices = {
                    pi: sol.component_choices
                    for pi, sol in joint_sol.port_solutions.items()
                }
                
                result = evaluate_joint_solution(
                    self.dut, port_comp_choices, freq_hz,
                    self.Z0, self.radiation_efficiency, self.per_port_efficiency,
                )
                
                if result.get('valid'):
                    all_mismatch.append(result['min_mismatch_efficiency'])
                    all_total.append(result['min_total_efficiency'])
        
        if all_mismatch:
            joint_sol.min_system_efficiency = float(np.min(all_mismatch))
            joint_sol.avg_system_efficiency = float(np.mean(all_mismatch))
            joint_sol.min_total_efficiency = float(np.min(all_total))
            joint_sol.avg_total_efficiency = float(np.mean(all_total))
            
            # Recompute unified score
            n_ports = max(
                max(joint_sol.port_metrics.keys(), default=-1) + 1,
                len(joint_sol.port_metrics),
            )
            avg_port_eff = []
            min_port_eff = []
            for pi in range(n_ports):
                pm = joint_sol.port_metrics.get(pi, {})
                eff = pm.get('total_efficiency', pm.get('mismatch_efficiency', 0))
                avg_port_eff.append(eff)
                min_port_eff.append(eff)
            
            all_effs = [pm.get('total_efficiency', pm.get('mismatch_efficiency', 0))
                        for pm in joint_sol.port_metrics.values()]
            
            score_input = ScoreInput(
                avg_port_efficiency=np.array(avg_port_eff),
                min_port_efficiency=np.array(min_port_eff),
                avg_band_efficiency=float(np.mean(all_total)) if all_total else 0.0,
                min_band_efficiency=float(np.min(all_total)) if all_total else 0.0,
                max_coupling_loss=joint_sol.max_coupling_loss,
                component_loss=joint_sol.component_loss_total,
                component_count=sum(
                    len(sol.component_choices)
                    for sol in joint_sol.port_solutions.values()
                ),
            )
            joint_sol.balanced_score = compute_unified_score(score_input, self.optimization_mode)
        
        return joint_sol
    
    def optimize(
        self,
        bands_mhz: Optional[List[List[float]]] = None,
        num_band_points: int = 5,
        progress_callback=None,
    ) -> List[JointSolution]:
        """
        Run the full joint multi-port optimization.
        
        Args:
            bands_mhz: Optional frequency bands for band-averaged evaluation
            num_band_points: Points per band
            progress_callback: Optional callback(progress_float, message_str)
        
        Returns:
            List of JointSolution sorted by balanced_score (best first)
        """
        start_time = time.time()
        self._debug_info = {}  # Reset debug info
        
        # Phase 1: Independent candidates
        if progress_callback:
            progress_callback(0.0, "Phase 1: Generating independent candidates per port...")
        
        candidates = self._phase1_independent_candidates()
        
        total_candidates = sum(len(v) for v in candidates.values())
        if progress_callback:
            progress_callback(0.4, f"Phase 1 done: {total_candidates} candidates across {len(candidates)} ports")
        
        # Collect phase1 debug info
        if self.debug:
            self._debug_info['phase1_candidates'] = {
                str(pi): [
                    {
                        'topology': sol.topology.name,
                        'rl_db': round(sol.s11_db, 2),
                        's11_mag': round(sol.s11_magnitude, 5),
                        'accepted_eff': round(1.0 - sol.s11_magnitude ** 2, 4),
                        'components': [
                            {
                                'connection_type': c.connection_type,
                                'value': f"{c.component.nominal_value}{c.component.nominal_unit}",
                                'part_number': c.component.part_number,
                            }
                            for c in sol.component_choices
                        ],
                    }
                    for sol in candidates[pi][:self.debug_top_n]
                ]
                for pi in candidates
            }
        
        # Phase 2: Joint evaluation
        if progress_callback:
            progress_callback(0.4, "Phase 2: Joint evaluation of all combinations...")
        
        joint_solutions = self._phase2_joint_evaluation(candidates)
        
        # Collect joint phase2 debug info
        if self.debug:
            self._debug_info['joint_top_candidates'] = [
                {
                    'score': round(sol.balanced_score, 5),
                    'avg_total_eff': round(sol.avg_total_efficiency, 5),
                    'min_total_eff': round(sol.min_total_efficiency, 5),
                    'max_coupling': round(sol.max_coupling_loss, 5),
                    'comp_loss': round(sol.component_loss_total, 5),
                    'ports': {
                        str(pi): {
                            'topology': sol.port_solutions[pi].topology.name,
                            'rl_db': round(sol.port_metrics.get(pi, {}).get('s11_db', 0), 2),
                            'total_eff': round(sol.port_metrics.get(pi, {}).get('total_efficiency', 0), 5),
                            'coupling': round(sol.port_metrics.get(pi, {}).get('coupling_loss', 0), 5),
                            'components': [
                                {
                                    'connection_type': c.connection_type,
                                    'value': f"{c.component.nominal_value}{c.component.nominal_unit}",
                                    'part_number': c.component.part_number,
                                }
                                for c in sol.port_solutions[pi].component_choices
                            ],
                        }
                        for pi in sol.port_solutions
                    },
                }
                for sol in joint_solutions[:min(self.debug_top_n, len(joint_solutions))]
            ]
        
        if progress_callback:
            progress_callback(0.7, f"Phase 2 done: {len(joint_solutions)} joint solutions evaluated")
        
        # Phase 3: Band evaluation (if configured)
        if bands_mhz and joint_solutions:
            if progress_callback:
                progress_callback(0.7, "Phase 3: Evaluating top solutions across frequency bands...")
            
            for i, js in enumerate(joint_solutions[:20]):  # Only top 20
                self._evaluate_at_band_frequencies(js, bands_mhz, num_band_points)
            
            # Re-sort after band evaluation
            joint_solutions.sort(key=lambda s: s.balanced_score, reverse=True)
        
        # Collect final ranking debug info
        if self.debug:
            self._debug_info['final_ranking'] = [
                {
                    'score': round(sol.balanced_score, 5),
                    'avg_total_eff': round(sol.avg_total_efficiency, 5),
                    'min_total_eff': round(sol.min_total_efficiency, 5),
                    'max_coupling': round(sol.max_coupling_loss, 5),
                    'comp_loss': round(sol.component_loss_total, 5),
                    'ports': {
                        str(pi): {
                            'topology': sol.port_solutions[pi].topology.name,
                            'rl_db': round(sol.port_metrics.get(pi, {}).get('s11_db', 0), 2),
                            'total_eff': round(sol.port_metrics.get(pi, {}).get('total_efficiency', 0), 5),
                            'coupling': round(sol.port_metrics.get(pi, {}).get('coupling_loss', 0), 5),
                            'components': [
                                {
                                    'connection_type': c.connection_type,
                                    'value': f"{c.component.nominal_value}{c.component.nominal_unit}",
                                    'part_number': c.component.part_number,
                                }
                                for c in sol.port_solutions[pi].component_choices
                            ],
                        }
                        for pi in sol.port_solutions
                    },
                }
                for sol in joint_solutions[:min(self.debug_top_n, len(joint_solutions))]
            ]
        
        elapsed = time.time() - start_time
        if progress_callback:
            progress_callback(1.0, f"Done in {elapsed:.1f}s. Best score: {joint_solutions[0].balanced_score:.4f}" if joint_solutions else "No valid solutions found")
        
        return joint_solutions
