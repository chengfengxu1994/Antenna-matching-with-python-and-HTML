# Optenni reference benchmark set

`cases.json` is the tracked manifest for Optenni parity work. The tutorial inputs remain in the licensed local Optenni installation and are not copied into this repository. Each implemented case now declares its evidence level, committed manifests/artifacts, whether it contains genuine cross-software numeric evidence, and the remaining gap. “Implemented” therefore means the product path exists; it does not automatically mean per-frequency Optenni parity has been proved.

All seven tracked workflows run through the core/product stack. Only `optimization-settings` currently has a native per-frequency Optenni curve export. Other cases are explicitly graded as saved-project/UI evidence, rounded tutorial references, or RFMatch-only recomputation. Golden CSV files exported by the user are stored under `benchmarks/optenni/golden/` only when their licensing and repository policy permit it.

Audit the committed evidence matrix without needing the licensed tutorial directory:

```powershell
python scripts/audit_optenni_evidence.py
```

Validate a user export before treating it as an oracle:

```powershell
$env:PYTHONPATH = (Resolve-Path "packages/rfmatch-core/src").Path
python scripts/validate_optenni_golden.py path/to/export.csv
```

Use `golden/optenni_matrix_template.csv` for new exports. It records the full
complex S matrix with unambiguous source/destination port semantics; the older
diagonal-only CSV format is still accepted for compatibility.
