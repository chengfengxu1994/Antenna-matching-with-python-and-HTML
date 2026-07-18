from concurrent.futures import ThreadPoolExecutor

import numpy as np

from engine.murata_db import MurataDatabase


def _database(path):
    database = MurataDatabase(str(path))
    database.connect()
    database.create_schema()
    cursor = database.conn.cursor()
    cursor.execute(
        "INSERT INTO series(id, name, manufacturer, component_type, size_code) "
        "VALUES (1, 'TEST', 'Vendor', 'inductor', '0402')"
    )
    cursor.execute(
        "INSERT INTO components(id, part_number, series_id, component_type, nominal_value, "
        "nominal_unit, tolerance_pct, s2p_filename, zip_path, is_primary) "
        "VALUES (1, 'L_TEST', 1, 'inductor', 5.6, 'nH', 5.0, 'L_TEST.s2p', 'test.zip', 1)"
    )
    for index, frequency in enumerate((1000.0, 2000.0), start=1):
        cursor.execute(
            "INSERT INTO freq_grid(id, freq_mhz, freq_hz) VALUES (?, ?, ?)",
            (index, frequency, frequency * 1e6),
        )
        cursor.execute(
            "INSERT INTO sparam_at_freq VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (1, index, 0.1 * index, 0.01, 0.9, -0.01, 0.9, -0.01, 0.1 * index, 0.01),
        )
        cursor.execute(
            "INSERT INTO derived_at_freq VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (1, index, 1.0, 2.0, 3.0, 4.0, 5.6, 'nH', 0.1, -20.0, 0.9, -0.9, 30.0, 5000.0),
        )
    database.conn.commit()
    return database


def test_runtime_database_reads_do_not_share_cursor_result_sets(tmp_path):
    database = _database(tmp_path / "components.db")

    def sparams_worker():
        for _ in range(150):
            result = database.get_component_sparams(1, 1500.0)
            assert np.isclose(result["s11"], complex(0.15, 0.01))

    def component_worker():
        for _ in range(150):
            records = database.get_primary_inductors()
            assert records[0].part_number == "L_TEST"

    def derived_worker():
        for _ in range(150):
            result = database.get_component_derived(1, 1500.0)
            assert np.isfinite(result["eff_value"])

    try:
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [
                executor.submit(worker)
                for worker in (sparams_worker, component_worker, derived_worker)
                for _ in range(4)
            ]
            for future in futures:
                future.result()
    finally:
        database.close()
