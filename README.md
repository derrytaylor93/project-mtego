# Project Mtego

Code accompanying the preregistration of the Project Mtego field experiment on snare detection by the Budongo Conservation Field Station snare patrol (Uganda).

Project Mtego ("snare" in Swahili) studies how snare detection depends on individual experience and how experience combines when patrol members search together. Groups search forest plots containing simulated snares, so the true number and location of snares is known.


## Contents

| Path | What it is |
|---|---|
| `simulations/pair_design_simulation.py` | Design simulation used to choose the number of solo trials per participant and snares per plot (see below). |

## Design simulation

`simulations/pair_design_simulation.py` is a planning tool, not the final power analysis. It:

1. invents 10 searchers with different abilities and 12 plots with fixed snare positions of varying difficulty;
2. simulates the design: each person searches alone *k* times, and all 45 pairs search once;
3. generates pair outcomes from the independent-searcher benchmark, P_pair = 1 − (1 − p_i)(1 − p_j), shifted on the log-odds scale by a synergy parameter δ;
4. fits the model to the simulated data and estimates δ;
5. repeats this many times to estimate the bias and standard error of δ and the power to detect a given synergy, for different numbers of snares per plot and solo trials per person.

All parameter values (between-person and between-snare variability, effect sizes) are assumptions and will be updated with pilot data.

Run it with (Python 3, NumPy, SciPy):

```
python simulations/pair_design_simulation.py 60   # 60 = simulations per design cell
```

## Data

No field data are stored in this repository. Snare and patrol locations are sensitive and must never be committed here. Any spatial data shared publicly will be aggregated or spatially masked.
