# CST result integration

RF Match Studio currently integrates CST results through validated Touchstone
exports. This boundary is intentional: the numerical engine consumes a
portable, auditable network representation instead of relying on CST's open
project state.

## Automatic export-folder synchronization

1. Export an `*.sNp` file from CST into the configured **SNP data directory**.
2. In **CST / HFSS results**, select `CST Studio Suite` as the source.
3. Click **Start watching**. Existing files form the baseline and are not
   reported as new results.
4. When CST creates or overwrites an export, RF Match Studio waits until file
   size and modification time remain unchanged for one second.
5. The stable file is parsed with the same strict Touchstone parser used by the
   optimizer. Only a complete port matrix with increasing frequencies is made
   available. With **Automatically load latest stable result** enabled, it
   becomes the active DUT immediately.

This prevents the common failure mode where an application reads a CST export
while the solver is still writing it. Changed files are delivered once per
revision; invalid completed exports remain visible as errors instead of
silently replacing the active DUT.

## API

- `POST /api/snp/watch/start?source=CST&stable_ms=1000`
- `GET /api/snp/watch/status?watch_id=...`
- `POST /api/snp/watch/stop?watch_id=...`

Watch sessions are in-memory and client-owned. They are bounded, require the
already configured SNP directory, and expose only relative files within that
directory.

## CST 2025 direct interface

The development machine currently contains CST Studio Suite 2025.0 and its
official `cst.interface` Python package. RF Match Studio detects the install
from the Windows uninstall registry, launches only CST's version-matched Python
runtime, and communicates through a small JSON worker. This keeps CST's Python
3.9 binary extension out of the product's Python 3.10 process.

`GET /api/cst/status` returns the exact runtime version, running Design
Environment process IDs and their open projects. `GET /api/cst/project-tree`
connects only to a project returned by that status call and reads its
S-parameter result nodes. `POST /api/cst/export-touchstone` invokes CST's
official `TOUCHSTONE` post-processing command, validates the generated `.sNp`,
stores it under a collision-safe project-derived name, and records SHA-bound
`cst_python_bridge` provenance. Direct export is unavailable while the solver
is running or before an S-parameter result exists. While a selected project is
solving, the web client refreshes its result tree every two seconds and unlocks
export automatically after CST reports completion.

The first direct export uses a readable project-derived filename. A later
export from the same canonical CST project atomically replaces that stable
filename and records `revision_of_sha256`; electrically identical content is reported as
unchanged instead of pretending to be a new revision. A same-name file without
matching CST provenance is never overwritten and receives a numbered sibling.
Detection does not start or close CST. When no Design Environment is running,
folder synchronization remains fully functional.

The live Windows baseline was verified against CST Studio Suite 2025.0 and an
already-open `test.cst`: two S-parameter result nodes were discovered, a 1001
point S1P was exported and loaded, and a repeat export reused `test.s1p`
without creating `test-2.s1p`.

CST may rewrite comments or formatting even when its numerical result did not
change, so byte SHA-256 remains the raw artifact audit identity but is not the
electrical revision test. RF Match Studio also computes
`rfmatch.touchstone-network.v1` over the canonical frequency vector, full
complex S-matrix and per-port reference impedances. When this network digest is
unchanged, the UI refreshes provenance while preserving port setup, manual
components, undo history, frozen variants and the backend optimization session.
A real numerical change produces
a different network digest and invalidates DUT-dependent working state.

## Source provenance

Uploads and stable watched exports are recorded in
`.rfmatch-snp-provenance.json` inside the configured SNP directory. Each record
contains the relative filename, source (`CST`, `HFSS`, `VNA`, etc.), ingestion
method, observation time and the exact Touchstone SHA-256. A record is returned
only while its digest matches the file contents; overwriting a result therefore
cannot silently inherit stale source metadata. The matching record follows the
active DUT into project snapshots and HTML/PDF reports.

Direct control never bypasses the stable Touchstone validation boundary. It
does not start a solver, modify model geometry, or guess which project to use;
the user explicitly selects an already-open project in the interface.
