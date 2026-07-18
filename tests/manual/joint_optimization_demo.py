"""Test joint multi-port optimization."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'apps', 'api'))

import numpy as np
from engine.touchstone import parse_touchstone
from engine.multiport_optimizer import JointMultiPortOptimizer, PortMatchConfig, evaluate_joint_solution
from engine.component_lib import scan_murata_directory
from engine.optimizer import MatchingOptimizer, OptimizerConfig, PortState
from project_paths import MURATA_DIR, SNP_DIR

# Load a 3-port antenna
with open(SNP_DIR / 'hfss_threeport_MA.s3p', 'r', encoding='utf-8', errors='replace') as f:
    dut = parse_touchstone(f.read(), 'hfss_threeport_MA.s3p')

print(f'Antenna: {dut.num_ports}-port, {len(dut.frequencies)} freq points')
print(f'Freq range: {min(dut.frequencies)/1e6:.1f} - {max(dut.frequencies)/1e6:.1f} MHz')

# Check raw coupling
mid_idx = len(dut.frequencies) // 2
S = dut.get_s_matrix(mid_idx)
freq_mhz = dut.frequencies[mid_idx] / 1e6
print(f'\nRaw S-matrix at {freq_mhz:.1f} MHz:')
for i in range(S.shape[0]):
    for j in range(S.shape[1]):
        mag = abs(S[i,j])
        ang = np.angle(S[i,j], deg=True)
        print(f'  S{i+1}{j+1}: {mag:.4f} < {ang:.1f} deg')

s21_db = 20*np.log10(max(abs(S[1,0]), 1e-15))
s31_db = 20*np.log10(max(abs(S[2,0]), 1e-15))
s32_db = 20*np.log10(max(abs(S[2,1]), 1e-15))
print(f'  |S21| = {s21_db:.1f} dB')
print(f'  |S31| = {s31_db:.1f} dB')
print(f'  |S32| = {s32_db:.1f} dB')

# Load component library
lib = scan_murata_directory(MURATA_DIR)
print(f'\nComponents: {len(lib.inductors)} inductors, {len(lib.capacitors)} capacitors')

# Run joint optimization with 2 ports
port_configs = [
    PortMatchConfig(port_index=0, max_components=2, target_frequency_hz=2.4e9),
    PortMatchConfig(port_index=1, max_components=2, target_frequency_hz=2.4e9),
]

optimizer = JointMultiPortOptimizer(
    dut=dut,
    component_library=lib,
    port_configs=port_configs,
    top_candidates_per_port=5,
    timeout_seconds=60,
    min_avg_balance=0.5,
)

joint_solutions = optimizer.optimize()
print(f'\nJoint solutions found: {len(joint_solutions)}')

if joint_solutions:
    best = joint_solutions[0]
    print(f'\n=== Best Joint Solution ===')
    print(f'Balanced score: {best.balanced_score:.4f}')
    print(f'Min system efficiency: {best.min_system_efficiency:.4f}')
    print(f'Avg system efficiency: {best.avg_system_efficiency:.4f}')
    print(f'Max coupling loss: {best.max_coupling_loss:.4f}')
    
    print(f'\nPort details:')
    for pi, metrics in best.port_metrics.items():
        s11_db = metrics['s11_db']
        mismatch = metrics['mismatch_efficiency']
        coupling = metrics['coupling_loss']
        radiated = metrics['radiated_efficiency']
        print(f'  Port {pi}: S11={s11_db:.1f}dB, '
              f'mismatch_eff={mismatch:.4f}, '
              f'coupling_loss={coupling:.4f}, '
              f'radiated_eff={radiated:.4f}')
    
    print(f'\nMatching components:')
    for pi, sol in best.port_solutions.items():
        comps = ', '.join(
            f'{c.connection_type}:{c.component.nominal_value}{c.component.nominal_unit}'
            for c in sol.component_choices
        )
        print(f'  Port {pi}: {sol.topology.name} -> {comps}')

    # === Comparison: Independent vs Joint ===
    print(f'\n=== Comparison: Independent vs Joint ===')
    
    # Independent: port 0 matched alone, port 1 at 50 ohm
    config = OptimizerConfig(
        target_frequency_hz=2.4e9,
        max_components=2,
        beam_width=5,
        timeout_seconds=20,
    )
    opt0 = MatchingOptimizer(dut, lib, config)
    sols0 = opt0.optimize_full(
        port_states={1: PortState.LOAD},
        input_port=0,
    )
    
    if sols0:
        ind_choices = {0: sols0[0].component_choices}
        ind_result = evaluate_joint_solution(dut, ind_choices, 2.4e9)
        if ind_result['valid']:
            p0 = ind_result['port_metrics'][0]
            p1 = ind_result['port_metrics'][1]
            print(f'Independent (port0 only):')
            print(f'  Port0 S11={p0["s11_db"]:.1f}dB, eff={p0["mismatch_efficiency"]:.4f}')
            print(f'  Port1 S11={p1["s11_db"]:.1f}dB, eff={p1["mismatch_efficiency"]:.4f}')
    
    # Joint: both ports matched together
    joint_choices = {
        pi: sol.component_choices for pi, sol in best.port_solutions.items()
    }
    joint_result = evaluate_joint_solution(dut, joint_choices, 2.4e9)
    if joint_result['valid']:
        p0 = joint_result['port_metrics'][0]
        p1 = joint_result['port_metrics'][1]
        print(f'Joint (both ports):')
        print(f'  Port0 S11={p0["s11_db"]:.1f}dB, eff={p0["mismatch_efficiency"]:.4f}')
        print(f'  Port1 S11={p1["s11_db"]:.1f}dB, eff={p1["mismatch_efficiency"]:.4f}')

print('\nJoint optimization test: OK')
