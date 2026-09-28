"""
Project Mtego - design simulation for the pair (synergy) test.
Rough planning tool, NOT a final power analysis. All parameter values are assumptions.

What it does, in plain terms
----------------------------
1. Invent 10 searchers, each with an ability (how good they are at spotting snares).
2. Invent 12 plots, each with a fixed set of snare positions; each snare has a difficulty.
3. Simulate the design: every person does k solo searches + all 45 pairs search once.
   Each trial happens in a randomly chosen plot.
   - Solo: P(find snare) = inv_logit(ability + snare_ease)
   - Pair, if they were just two independent searchers:
         P_indep = 1 - (1 - p_i) * (1 - p_j)       (found if EITHER would find it)
     Synergy is added on the log-odds scale:
         P_pair = inv_logit( logit(P_indep) + delta )
     delta = 0 -> no synergy; delta > 0 -> synergy; delta < 0 -> interference.
4. Fit the same model to the simulated data (maximum likelihood) and estimate delta.
5. Repeat many times: how much does the estimate of delta bounce around (its SE),
   and how often would we detect a given true delta (power)?

Converting delta to percentage points (pp)
------------------------------------------
If two people each find 50% alone, P_indep = 1 - 0.5*0.5 = 0.75.
   delta = 0.5 -> inv_logit(logit(0.75) + 0.5) = 0.83  -> +8 pp above benchmark
   delta = 1.0 -> inv_logit(logit(0.75) + 1.0) = 0.89  -> +14 pp
These delta values were chosen by me for illustration, not derived from any data.
"""
import itertools
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit as inv_logit, logit

rng = np.random.default_rng(1)

# ---------------- ASSUMPTIONS (all invented; change freely) ----------------
N_PEOPLE = 10
N_PLOTS = 12
ABILITY_MEAN = 0.0   # log-odds; 0.0 -> average solo detection ~50%, 0.9 -> ~68%
ABILITY_SD = 0.7     # how much people differ from one another
SNARE_SD = 1.0       # how much snare positions differ in difficulty
PAIRS = list(itertools.combinations(range(N_PEOPLE), 2))   # 45 pairs


def simulate(n_snares, k_solo, delta, ability_mean=ABILITY_MEAN):
    """Return one simulated dataset: one row per (trial, snare)."""
    ability = rng.normal(ability_mean, ABILITY_SD, N_PEOPLE)
    snare_ease = rng.normal(0, SNARE_SD, (N_PLOTS, n_snares))
    trials = [(i,) for i in range(N_PEOPLE) for _ in range(k_solo)] + PAIRS
    rows = []
    for t, members in enumerate(trials):
        plot = rng.integers(N_PLOTS)
        for s in range(n_snares):
            p_each = inv_logit(ability[list(members)] + snare_ease[plot, s])
            if len(members) == 1:
                p = p_each[0]
            else:
                p_indep = 1 - (1 - p_each[0]) * (1 - p_each[1])
                p = inv_logit(logit(p_indep) + delta)
            found = rng.random() < p
            j = members[1] if len(members) > 1 else -1
            rows.append((members[0], j, plot * n_snares + s, found, t))
    return np.array(rows, dtype=int)


def estimate_delta(data, n_snares):
    """Maximum-likelihood fit of abilities, snare difficulties and delta.
    Snare difficulties get a mild penalty (like a random effect with SD 1)."""
    i, j, s, y = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    n_snare_ids = N_PLOTS * n_snares
    is_pair = j >= 0
    jj = np.where(is_pair, j, 0)

    def neg_loglik(theta):
        a = theta[:N_PEOPLE]
        b = theta[N_PEOPLE:N_PEOPLE + n_snare_ids]
        d = theta[-1]
        qi = inv_logit(a[i] + b[s])
        qj = inv_logit(a[jj] + b[s])
        u = np.clip(np.where(is_pair, 1 - (1 - qi) * (1 - qj), qi), 1e-9, 1 - 1e-9)
        eta = np.where(is_pair, logit(u) + d, a[i] + b[s])
        p = inv_logit(eta)
        ll = np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)) - 0.5 * np.sum(b ** 2)
        # analytic gradient
        r = y - p
        g = np.zeros_like(theta)
        np.add.at(g, i, r * np.where(is_pair, qi / u, 1.0))
        np.add.at(g, jj, r * np.where(is_pair, qj / u, 0.0))
        np.add.at(g[N_PEOPLE:N_PEOPLE + n_snare_ids], s, r * np.where(is_pair, (qi + qj) / u, 1.0))
        g[N_PEOPLE:N_PEOPLE + n_snare_ids] -= b
        g[-1] = np.sum(r * is_pair)
        return -ll, -g

    start = np.zeros(N_PEOPLE + n_snare_ids + 1)
    return minimize(neg_loglik, start, jac=True, method="L-BFGS-B").x[-1]


if __name__ == "__main__":
    n_sim = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    for n_snares in [5, 10, 15, 20]:
        for k_solo in [1, 2, 3]:
            null = np.array([estimate_delta(simulate(n_snares, k_solo, 0.0), n_snares)
                             for _ in range(n_sim)])
            se = null.std()
            # NOTE: detection threshold is set around the mean of the null estimates,
            # i.e. it corrects for the estimator's bias. This flatters power somewhat.
            crit = 1.96 * se
            power = {}
            for delta in [0.5, 1.0]:
                est = np.array([estimate_delta(simulate(n_snares, k_solo, delta), n_snares)
                                for _ in range(n_sim // 2)])
                power[delta] = np.mean(np.abs(est - null.mean()) > crit)
            print(f"snares={n_snares:2d}  solo_trials={k_solo}  "
                  f"bias_under_null={null.mean():+.2f}  SE={se:.2f}  "
                  f"power(+8pp)={power[0.5]:.2f}  power(+14pp)={power[1.0]:.2f}")
