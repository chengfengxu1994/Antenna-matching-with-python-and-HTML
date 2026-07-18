# Optenni reference cases

The project reads tutorial data from an external installation and does not copy proprietary tutorial material.

Initial cases:

- Quick Start: `measured_antenna.s1p`, Band 7, two-element L networks.
- Optimization Settings: minimum/average weighting and continuous-to-discrete workflow.
- Multiantenna: complete coupled-matrix evaluation is a mandatory design constraint.
- Radiation Efficiency: B8/B4/B40 and frequency-dependent radiation efficiency.

The tutorial screenshots are descriptive references, not machine-readable golden results. For strict numerical comparison, export candidate topology, component values and per-frequency efficiency from Optenni into a future `golden/` JSON file. The regression runner will compare those exports without invoking or reverse-engineering Optenni Lab.
