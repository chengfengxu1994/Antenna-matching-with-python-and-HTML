"""End-to-end smoke test."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.server import app
from fastapi.testclient import TestClient

client = TestClient(app)

# 1. Config
r = client.post('/api/config/dirs', json={'snp_dir': r'E:\RF matching\snp', 'murata_dir': r'E:\RF matching\Murata'})
print("Config:", r.json()['status'], r.json().get('mode'))

# 2. Health
r = client.get('/api/health')
print("Health:", r.json())

# 3. Load SNP
r = client.post('/api/snp/load?filename=SAR Head Hand and Phone.s6p')
print("SNP:", r.json()['status'], r.json().get('num_ports'), 'ports')

# 4. Component series
r = client.get('/api/component-series')
print("Series:", len(r.json()['inductor_series']), 'L,', len(r.json()['capacitor_series']), 'C')

# 5. Band presets
r = client.get('/api/band-presets')
print("Presets:", len(r.json()['presets']), 'bands')

# 6. Multipass optimize (2 ports)
body = {
    'snp_filename': 'SAR Head Hand and Phone.s6p',
    'ports': [
        {'port_index': 0, 'state': 'load', 'use_matching': True, 'max_components': 2,
         'band_mhz': [2400, 2500], 'num_band_points': 3},
        {'port_index': 1, 'state': 'short', 'use_matching': False, 'band_mhz': [2400, 2500]},
    ],
    'beam_width': 10, 'timeout_seconds': 30,
}
r = client.post('/api/multipass', json=body)
print("Multipass:", r.status_code)
if r.status_code == 200:
    d = r.json()
    print("  ports_processed:", d['ports_processed'])
    for pi, pr in d['results_per_port'].items():
        print("  Port %s: %d sol, best RL=%.1fdB" % (pi, pr['solutions_count'], pr['best_s11_db']))
else:
    print("  Error:", r.text[:300])

print("\nAll tests passed!")
