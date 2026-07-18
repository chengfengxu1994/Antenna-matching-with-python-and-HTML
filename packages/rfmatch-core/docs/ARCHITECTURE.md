# Architecture

The kernel is intentionally independent from the existing FastAPI and React application.

## Numerical contract

1. Port indices always refer to the original, unreduced network.
2. Reference impedance is explicit in every S/Z/Y operation.
3. A candidate is evaluated on every configured frequency and on the complete multiport matrix.
4. Total efficiency is radiation efficiency multiplied by power not reflected or coupled to another port.
5. Component dissipation is not guessed. Ideal elements are lossless; measured two-port models will be added through an explicit power-wave implementation.

## Search contract

The first implementation follows the public workflow described in the Optenni 4.3 optimization tutorial:

- deterministic multi-start continuous search;
- local mapping into discrete component values;
- complete objective evaluation at every search comparison;
- separate minimum/average controls within a band, across bands, and across ports;
- optional impedance target and component-count penalty.

No partial-network S11 beam pruning is used. Future search algorithms must pass the same evaluator and regression suite.

## Long-term extension points

- measured S2P component model and exact dissipated-power accounting;
- topology grammar with physical node ordering;
- tolerance Monte Carlo and yield ranking;
- simultaneous multiport discrete refinement;
- branch-and-bound bounds that are mathematically valid for the complete objective;
- versioned benchmark result files for comparison with manually exported Optenni results.

## Physical topology syntax

`CircuitTopology` separates measurement reference-plane nodes (`external_nodes`)
from DUT terminal nodes (`dut_nodes`). `Branch` connects named physical nodes
using either `LumpedModel` or a measured `S2PModel`. Internal nodes are eliminated
only after the complete nodal admittance matrix is assembled.

For each unit incident power wave the solver reconstructs all node voltages. The
loss of each component is calculated as `real(V.H @ Y_component @ V)`. Reflected
power, every component loss and DUT absorbed power are checked against unity.

S2P tolerance is deliberately not implemented by scaling S-parameters. Use
measured min/nominal/max corner files for S2P tolerance work; scalar tolerance is
supported for physical L/C/R values.
