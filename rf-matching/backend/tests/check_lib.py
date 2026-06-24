import sys, os; sys.path.insert(0, '.')
from engine.component_lib import scan_murata_directory
lib = scan_murata_directory(r'E:\RF matching\Murata')

lqp = [c for c in lib.inductors if 'LQP03HQ' in c.part_number.upper()]
vals_l = sorted(set(c.nominal_value for c in lqp))
print(f'LQP03HQ: {len(lqp)} parts, {len(vals_l)} unique values')
print(f'  Range: {vals_l[0]:.2f} ~ {vals_l[-1]:.2f} nH')
print(f'  Values: {[round(v,2) for v in vals_l]}')

gjm = [c for c in lib.capacitors if 'GJM03' in c.part_number.upper()]
vals_c = sorted(set(c.nominal_value for c in gjm if c.nominal_value > 0.01))
print(f'\nGJM03: {len(gjm)} parts, {len(vals_c)} unique values')
print(f'  Range: {vals_c[0]:.2f} ~ {vals_c[-1]:.2f} pF')
print(f'  Values: {[round(v,2) for v in vals_c]}')
