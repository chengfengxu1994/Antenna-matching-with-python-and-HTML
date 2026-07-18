"""Manually invoked end-to-end smoke test.

This module deliberately performs no HTTP work when pytest imports it during
test discovery. Run it directly when the local reference data is available.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "apps", "api"))


def main() -> int:
    from api.server import app
    from fastapi.testclient import TestClient
    from project_paths import MURATA_DIR, SNP_DIR

    client = TestClient(app)

    response = client.post(
        "/api/config/dirs",
        json={"snp_dir": str(SNP_DIR), "murata_dir": str(MURATA_DIR)},
    )
    print("Config:", response.json()["status"], response.json().get("mode"))
    print("Health:", client.get("/api/health").json())

    response = client.post("/api/snp/load?filename=SAR Head Hand and Phone.s6p")
    response.raise_for_status()
    payload = response.json()
    print("SNP:", payload.get("status", "loaded"), payload.get("num_ports"), "ports")

    payload = client.get("/api/component-series").json()
    print("Series:", len(payload["inductor_series"]), "L,", len(payload["capacitor_series"]), "C")
    print("Presets:", len(client.get("/api/band-presets").json()["presets"]), "bands")

    body = {
        "snp_filename": "SAR Head Hand and Phone.s6p",
        "ports": [
            {
                "port_index": 0, "state": "load", "use_matching": True,
                "max_components": 2, "band_mhz": [2400, 2500], "num_band_points": 3,
            },
            {
                "port_index": 1, "state": "short", "use_matching": False,
                "band_mhz": [2400, 2500],
            },
        ],
        "beam_width": 10,
        "timeout_seconds": 30,
    }
    response = client.post("/api/multipass", json=body)
    print("Multipass:", response.status_code)
    if response.status_code != 200:
        print("  Error:", response.text[:300])
        return 1
    payload = response.json()
    print("  ports_processed:", payload["ports_processed"])
    for port_index, port_result in payload["results_per_port"].items():
        print(
            "  Port %s: %d sol, best RL=%.1fdB"
            % (port_index, port_result["solutions_count"], port_result["best_s11_db"])
        )
    print("\nAll tests passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
