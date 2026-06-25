"""
Power balance analysis for RF matching/tuning systems.

Computes the full power flow breakdown per port and for the whole system:

    P_incident  →  P_reflected  (mismatch)
                 + P_coupled    (transferred to other ports)
                 + P_component_loss (dissipated in matching components)
                 + P_antenna_loss  (ohmic loss in antenna, from η_rad)
                 + P_radiated     (actually radiated into free space)

All values are normalized so that P_incident = 1.0 (fractional power).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class PortPowerBalance:
    """Power balance for a single port at a single frequency."""
    # Input
    incident: float = 1.0
    # Loss mechanisms (fraction of incident power)
    reflected: float = 0.0      # |Sii|²
    coupled: float = 0.0        # Σ|Sji|² for j≠i
    component_loss: float = 0.0 # dissipated in matching components
    antenna_loss: float = 0.0   # ohmic loss in antenna itself
    # Output
    radiated: float = 0.0       # actually radiated

    @property
    def accepted(self) -> float:
        """Power accepted into the port: 1 - reflected."""
        return 1.0 - self.reflected

    @property
    def total_loss(self) -> float:
        """Total power lost: reflected + coupled + component_loss + antenna_loss."""
        return self.reflected + self.coupled + self.component_loss + self.antenna_loss

    @property
    def sum_check(self) -> float:
        """Check power conservation: should be 1.0."""
        return self.total_loss + self.radiated

    def to_dict(self) -> dict:
        return {
            'incident': float(self.incident),
            'reflected': float(self.reflected),
            'coupled': float(self.coupled),
            'component_loss': float(self.component_loss),
            'antenna_loss': float(self.antenna_loss),
            'radiated': float(self.radiated),
            'accepted': float(self.accepted),
            'sum_check': float(self.sum_check),
        }


@dataclass
class SystemPowerBalance:
    """Power balance across all ports."""
    per_port: Dict[int, PortPowerBalance] = field(default_factory=dict)
    total_incident: float = 0.0
    total_reflected: float = 0.0
    total_coupled: float = 0.0
    total_component_loss: float = 0.0
    total_antenna_loss: float = 0.0
    total_radiated: float = 0.0

    @property
    def system_efficiency(self) -> float:
        """Fraction of total incident power that is radiated."""
        if self.total_incident > 0:
            return self.total_radiated / self.total_incident
        return 0.0

    def to_dict(self) -> dict:
        return {
            'per_port': {str(k): v.to_dict() for k, v in self.per_port.items()},
            'total_incident': float(self.total_incident),
            'total_reflected': float(self.total_reflected),
            'total_coupled': float(self.total_coupled),
            'total_component_loss': float(self.total_component_loss),
            'total_antenna_loss': float(self.total_antenna_loss),
            'total_radiated': float(self.total_radiated),
            'system_efficiency': float(self.system_efficiency),
        }


def compute_power_balance(
    s_matrix: np.ndarray,
    component_loss_total: float = 0.0,
    matched_ports: Optional[List[int]] = None,
    radiation_efficiency: Optional[Dict[int, float]] = None,
    n_matched_ports: int = 0,
) -> SystemPowerBalance:
    """
    Compute power balance for all ports from the full system S-matrix.

    Args:
        s_matrix: NxN S-matrix AFTER all matching networks applied
        component_loss_total: Total fractional power lost in all matching components
        matched_ports: Which ports have matching networks (for allocating component loss)
        radiation_efficiency: Per-port antenna radiation efficiency {port_index: eta_rad}
        n_matched_ports: Fallback count if matched_ports not provided

    Returns:
        SystemPowerBalance with per-port breakdown
    """
    n = s_matrix.shape[0]
    if matched_ports is None:
        n_matched = n_matched_ports or max(1, n)
    else:
        n_matched = len(matched_ports)

    pb = SystemPowerBalance()
    pb.total_incident = float(n)  # Each port has unit incident power

    for i in range(n):
        sii = s_matrix[i, i]
        reflected = abs(sii) ** 2
        coupled = sum(abs(s_matrix[j, i]) ** 2 for j in range(n) if j != i)

        # Allocate component loss: only for ports that have matching
        if matched_ports and i in matched_ports:
            comp_loss = component_loss_total / max(n_matched, 1)
        elif not matched_ports:
            comp_loss = component_loss_total / max(n_matched, 1)
        else:
            comp_loss = 0.0

        # Antenna ohmic loss
        eta_rad = 1.0
        if radiation_efficiency and i in radiation_efficiency:
            eta_rad = radiation_efficiency[i]
        accepted = 1.0 - reflected
        radiated_from_accepted = max(0.0, accepted - coupled - comp_loss)
        ant_loss = radiated_from_accepted * (1.0 - eta_rad) if eta_rad < 1.0 else 0.0
        radiated = radiated_from_accepted * eta_rad

        port_pb = PortPowerBalance(
            incident=1.0,
            reflected=float(reflected),
            coupled=float(coupled),
            component_loss=float(comp_loss),
            antenna_loss=float(ant_loss),
            radiated=float(radiated),
        )
        pb.per_port[i] = port_pb
        pb.total_reflected += port_pb.reflected
        pb.total_coupled += port_pb.coupled
        pb.total_component_loss += port_pb.component_loss
        pb.total_antenna_loss += port_pb.antenna_loss
        pb.total_radiated += port_pb.radiated

    return pb


def power_balance_to_chart_data(pb: SystemPowerBalance) -> List[dict]:
    """
    Convert power balance to chart-friendly format (stacked bar chart).

    Returns:
        List of dicts, one per port:
        {port, reflected, coupled, component_loss, antenna_loss, radiated}
    """
    data = []
    for port_idx in sorted(pb.per_port.keys()):
        p = pb.per_port[port_idx]
        data.append({
            'port': f'Port {port_idx + 1}',
            'port_index': port_idx,
            'reflected': p.reflected * 100,
            'coupled': p.coupled * 100,
            'component_loss': p.component_loss * 100,
            'antenna_loss': p.antenna_loss * 100,
            'radiated': p.radiated * 100,
            'efficiency_pct': p.radiated * 100,
        })
    return data


def power_balance_from_evaluation(
    eval_result: dict,
    n_ports: int,
) -> SystemPowerBalance:
    """
    Compute power balance from evaluate_joint_solution() output.

    Args:
        eval_result: The dict returned by evaluate_joint_solution()
        n_ports: Number of DUT ports

    Returns:
        SystemPowerBalance
    """
    s_matrix = eval_result.get('s_matrix')
    if s_matrix is None:
        raise ValueError("evaluation result must contain 's_matrix'")

    comp_loss = eval_result.get('component_loss_total', 0.0)
    power_balance_dict = eval_result.get('power_balance', {})

    # Determine which ports are matched (those in power_balance with component_loss > 0)
    matched_ports = [
        int(k) for k, v in power_balance_dict.items()
        if v.get('component_loss', 0) > 0
    ]

    return compute_power_balance(
        s_matrix=s_matrix,
        component_loss_total=comp_loss,
        matched_ports=matched_ports if matched_ports else None,
        n_matched_ports=len(eval_result.get('port_configs', {})),
    )
