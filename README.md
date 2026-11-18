# Project Mtego

Code accompanying the preregistration of the Project Mtego field experiment on snare detection by the Budongo Conservation Field Station snare patrol (Uganda).

Project Mtego ("snare" in Swahili) studies how snare detection depends on individual experience and how experience combines when patrol members search together. Groups search forest plots containing simulated snares, so the true number and location of snares is known.


## Contents

| Path | What it is |
|---|---|
| `simulations/pair_design_simulation.py` | First design simulation, used to choose the number of solo trials per participant and snares per plot (see below). |
| `simulations/design_simulation_v2.py` | Second design simulation: one plot per field day, all three research questions, 2 vs 3 solo trials (see below). |
| `simulations/summarise_v2.py` | Prints the summary table from a v2 results file. |
| `simulations/results/` | Results of the v2 run (400 simulated studies per scenario) and its printed summary. |

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

## Design simulation, version 2

`simulations/design_simulation_v2.py` updates the simulation to the current design and checks all three research questions, not only synergy:

* **Schedule:** one plot per field day. Every participant searches every plot once, so each day has 2 solos + 4 pairs, 4 solos + 3 pairs, or (with 2 solo trials per person) 5 pairs. Snares are the same for everyone within a day.
* **Experience:** ability depends on years of experience. The true curve is flat, linear or levelling off.
* **H1:** recovery of the experience model by leave-one-participant-out cross-validation, using the preregistered rule (the winning model must beat the next by more than 2 SE). It also reports precision of the slope and of E90.
* **H2:** recovery of the pair benchmark (pooled experience, best member, independent searchers) by 5-fold cross-validation over pairs.
* **Synergy (δ):** bias and precision of δ against the independent-searcher benchmark and against the pooled-experience prediction.

It compares 2 and 3 solo trials per participant, with 20 snares per plot. There is no trial-level (good day / bad day) variation and no position-in-day effect. Models are fitted by penalised maximum likelihood, using the preregistered priors as penalties; this is a fast stand-in for the Bayesian models. As before, all parameter values are assumptions.

```
OMP_NUM_THREADS=1 python simulations/design_simulation_v2.py 400 400 4 results.json   # reps H1, reps H2, workers, output
python simulations/summarise_v2.py results.json
```

Set `OMP_NUM_THREADS=1` when using several workers, or the numerical libraries oversubscribe the CPU. With 400 repetitions and 4 workers the run takes about 6 minutes.

Main results (`simulations/results/design_simulation_v2_summary.txt`):

* **H1:** with 10 participants the 2-SE rule picks a model in only 3–9% of studies, whether there are 2 or 3 solo trials. The size of the experience effect can be estimated (slope SE ≈ 0.25 log-odds per 10 years), but E90 cannot.
* **H2:** the best-scoring benchmark is the true one in 70–91% of studies, but the 2-SE rule confirms it in only 13–38%. It almost never picks the wrong one. A third solo trial adds about 2–4 points.
* **δ against independent searchers:** SE 0.17 (2 solos) vs 0.15 (3 solos). The fit underestimates δ by about 0.25. After correcting for this, δ = 0.5 is detected 75% vs 88% of the time.
* **δ against pooled experience:** SE 0.22 with either 2 or 3 solo trials, and no bias. δ = 0.5 is detected about 50% of the time.

## Data

No field data are stored in this repository. Snare and patrol locations are sensitive and must never be committed here. Any spatial data shared publicly will be aggregated or spatially masked.
