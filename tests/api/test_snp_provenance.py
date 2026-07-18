import asyncio
import hashlib
import json
from pathlib import Path
import tempfile

from api import server
from api.models import SNPImportRequest
from engine.snp_provenance import SnpProvenanceStore, STORE_FILENAME


CONTENT = "# GHZ S RI R 50\n1.0 0.1 0\n1.1 0.2 0\n"


def test_provenance_is_content_bound_and_atomically_persisted(tmp_path: Path):
    source = tmp_path / "antenna.s1p"
    source.write_text(CONTENT, encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    store = SnpProvenanceStore()

    recorded = store.record(
        tmp_path, "antenna.s1p", sha256=digest, source="CST",
        ingestion_method="directory_watch", details={"project": "antenna.cst"},
    )
    assert recorded["source"] == "CST"
    assert store.lookup(tmp_path, "antenna.s1p", sha256=digest) == recorded
    assert not list(tmp_path.glob("*.tmp"))
    document = json.loads((tmp_path / STORE_FILENAME).read_text(encoding="utf-8"))
    assert document["schema_version"] == 1

    changed_digest = hashlib.sha256((CONTENT + "! changed\n").encode()).hexdigest()
    assert store.lookup(tmp_path, "antenna.s1p", sha256=changed_digest) is None


def test_import_and_load_round_trip_preserves_source_provenance():
    original = (
        server.state.snp_dir, server.state.loaded_snp,
        server.state.loaded_snp_filename, server.state.loaded_snp_provenance,
    )
    try:
        with tempfile.TemporaryDirectory() as directory:
            server.state.snp_dir = directory
            imported = asyncio.run(server.import_snp(SNPImportRequest(
                filename="antenna.s1p", content=CONTENT, source="HFSS",
            )))
            assert imported["provenance"]["source"] == "HFSS"
            assert imported["provenance"]["ingestion_method"] == "file_import"

            loaded = asyncio.run(server.load_snp(imported["filename"]))
            assert loaded["provenance"]["source"] == "HFSS"
            assert server.state.loaded_snp_provenance["sha256"] == imported["sha256"]
    finally:
        (
            server.state.snp_dir, server.state.loaded_snp,
            server.state.loaded_snp_filename, server.state.loaded_snp_provenance,
        ) = original

