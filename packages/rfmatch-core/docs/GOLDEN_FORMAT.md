# Golden Data Format — Optenni Export Schema

This format is the numerical contract between Optenni Lab exports and
`rfmatch-core`. New multiport references use the complete-matrix schema. The
original diagonal-only schema remains readable so existing baselines do not
break.

## Canonical complete-matrix schema

| Column | Required | Units | Meaning |
|---|---:|---|---|
| `frequency_hz` | yes | Hz | Positive frequency. |
| `source_port` | yes | — | 1-based driven/incident port, j in Sij. |
| `destination_port` | yes | — | 1-based observed/output port, i in Sij. |
| `s_real` | yes | linear | Real part of Sij. |
| `s_imag` | yes | linear | Imaginary part of Sij. |
| `total_efficiency` | no | linear power fraction | Efficiency for the driven `source_port`. |
| `component_loss` | no | linear power fraction | Matching-network dissipation for the driven `source_port` and unit incident power. |

The convention is explicit: a row with `source_port=1` and
`destination_port=2` contains **S21**. Internally, NumPy S-parameter arrays use
`s[frequency, destination_port, source_port]`.

Use one row per `(frequency_hz, source_port, destination_port)`. For an N-port
complete matrix, each frequency therefore has N² rows. Power metrics belong to
the source port. Put them on the diagonal row for that source, or repeat the
same value on all of its destination rows; repeated values must be identical.

Rows may be grouped by Sij pair or emitted in frequency-major order, provided
frequencies increase strictly within each pair.

## Legacy diagonal schema

The following columns remain supported:

`frequency_hz,port,s11_real,s11_imag,total_efficiency,component_loss`

For this schema, `port=2` means the complex columns contain S22, not S11. It
cannot validate coupling terms and should not be used for new multiport golden
files.

## Validation and comparison

Run:

```powershell
python scripts/validate_optenni_golden.py benchmarks/optenni/golden/example.csv
```

The summary reports covered ports and Sij pairs, whether every frequency has a
complete square S matrix, and which optional power metrics are present. The
comparison API reports complex magnitude error `abs(computed - reference)` for
every Sij sample plus absolute efficiency and loss errors.

Golden data must come from a licensed Optenni Lab session and should include
the original Touchstone input, exact circuit topology/BOM, optimization
settings, Optenni version, and export date in the case metadata. Do not treat
plots or screenshots as numerical ground truth.
