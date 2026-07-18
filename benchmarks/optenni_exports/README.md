# Optenni exported baselines

This directory stores user-authorized numeric exports from Optenni Lab for
point-by-point cross-validation. Keep the original filename, selected topology,
component values, frequency configuration, Optenni version, and export date in
the adjacent manifest before treating a file as a golden reference.

`quick_start_0402cs_gjm15_pcsl_plot.txt` and its adjacent circuit S2P are the
native 531-point Optenni 4.3 Quick Start result for 2500–2690 MHz, rebuilt with
the official Coilcraft 0402CS and Murata GJM15 families.  Terminating exported
network port 2 in the original DUT and observing port 1 reproduces every native
plot row with at most 2.42e-4 dB S11 error and 5.92e-5 dB total-efficiency
error.  Run `python scripts/validate_optenni_native_export.py` to verify file
hashes, port orientation, passive transducer efficiency and all 531 points.

`optimization_settings_pcsl_tolerance_100.txt` is a native Optenni 4.3
tolerance plot export. It contains 531 frequency rows, one nominal curve and
100 Monte Carlo variants for both S11 and total efficiency. Optenni repeats the
same header for every variant, so it must be read by column position using
`rfmatch_core.load_optenni_tolerance_export`, not by a dictionary CSV reader.

For the 1.7–2.5 GHz target, Optenni's displayed 0% yield is reproduced by
jointly applying minimum total efficiency `-1.0 dB` and dB-domain arithmetic
average total efficiency `-0.7 dB`. The minimum-efficiency criterion alone
passes 54 of 100 samples. This is a geometric mean in linear units. Exact
settings, statistics and SHA-256 are recorded in the manifest.

`optimization_settings_4bands_default_best_plot.txt` is a second native
531-point export from the same project. It covers four optimization bands and
the six-element `PLSCSLPCSLPC` candidate. Its UI values are rounded; the
adjacent manifest records values fitted from the complete exported S11 curve.
