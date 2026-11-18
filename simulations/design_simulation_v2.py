"""
Project Mtego - design simulation, version 2 (one plot per day).

Rough planning tool, NOT a final power analysis. All parameter values are assumptions.

What is new compared with pair_design_simulation.py
---------------------------------------------------
* Trials follow the one-plot-per-day schedule: every participant searches every plot
  once, so each plot (= field day) has 2 solos + 4 pairs, 4 solos + 3 pairs, or
  (with 2 solo trials) 5 pairs. Snares are the same for everyone within a day.
* Ability depends on experience (H1 truth: none, linear or levelling off).
* It evaluates all three questions, comparing 2 vs 3 solo trials per person:
    H1  - which experience model is recovered (leave-one-participant-out CV,
          preregistered rule: winner must beat the next model by > 2 SE), and
          how precisely the experience effect is estimated.
    H2  - which pair benchmark is recovered (pooled experience, best member,
          independent searchers), by cross-validation over pairs.
    H3  - precision of delta against the independent-searchers benchmark (delta_c)
          and against the pooled-experience benchmark (delta_a).
* No trial-level (good day / bad day) variation and no position-in-day effect.

Fitting is penalised maximum likelihood with the preregistered priors as penalties
(a fast stand-in for the Bayesian brms models). In H1, each participant's own
deviation from the experience curve is integrated out (Gauss-Hermite quadrature), so
its SD is estimated. In H2 that SD is fixed at the H1 estimate from the same dataset.
H2 uses 5-fold cross-validation over pairs as a cheaper stand-in for leave-one-pair-out.

Usage: python mtego_design_sim_v2.py [n_reps_H1] [n_reps_H2] [n_workers]
"""
import itertools
import sys
import random
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

N = 10
N_SNARES = 20
SNARE_SD = 1.0
E_RANGE = (2.0, 30.0)          # years of experience (patrol + hunting)
SIGMA_V = 0.5                  # person deviation from the experience curve (log-odds)

# H1 truths (log-odds ability as a function of experience E in years)
LIN = dict(lo=-0.8, hi=0.8)                   # straight line from -0.8 (E=2) to +0.8 (E=30)
ASYM = dict(A=0.7, A0=-1.5, k=0.15)           # E90 = ln(10)/0.15 = 15.4 years
M0_SD = 0.7                                   # no experience effect: ability SD 0.7

GH_Z, GH_W = np.polynomial.hermite_e.hermegauss(15)
GH_LW = np.log(GH_W / np.sqrt(2 * np.pi))


# ---------------------------------------------------------------- schedules
def round_robin(n=N):
    ps = list(range(n)); R = []
    for _ in range(n - 1):
        R.append([(ps[i], ps[n - 1 - i]) for i in range(n // 2)])
        ps = [ps[0]] + [ps[-1]] + ps[1:-1]
    return R


def make_schedule(k_solo, seed=0):
    """List of days; each day = (solo list, pair list). Every person once per day."""
    rnd = random.Random(seed)
    R = round_robin()
    for _ in range(500000):
        removed = [rnd.choice(r) for r in R]
        base = [([p for p in r if p != rm], list(rm)) for r, rm in zip(R, removed)]
        solo = [0] * N
        for _, so in base:
            for x in so: solo[x] += 1
        if k_solo == 3:
            perm = rnd.sample(removed, 9); groups = [perm[0:3], perm[3:6], perm[6:9]]
        else:  # 10 days with 2 solos + 4 pairs, 1 day with 5 pairs
            perm = rnd.sample(removed, 9); groups = [perm[0:4], perm[4:9]]
        if not all(len({x for p in g for x in p}) == 2 * len(g) for g in groups):
            continue
        extra = []
        s = solo[:]
        for g in groups:
            inp = {x for p in g for x in p}
            so = [x for x in range(N) if x not in inp]
            for x in so: s[x] += 1
            extra.append((g, so))
        if all(v == k_solo for v in s):
            days = [(pr, so) for pr, so in base] + extra
            return days
    raise RuntimeError("no schedule found")


SCHED = {3: make_schedule(3), 2: make_schedule(2)}


# ---------------------------------------------------------------- curves
def curve(kind, th, E, Emean, Esd):
    """Return g (len E) and Jacobian dg/dtheta (len E x n_theta)."""
    if kind == "M0":
        return np.full(len(E), th[0]), np.ones((len(E), 1))
    if kind == "M1a":
        z = (E - Emean) / Esd
        return th[0] + th[1] * z, np.column_stack([np.ones(len(E)), z])
    if kind == "M1b":
        A, A0, lk = th; k = np.exp(lk); e = np.exp(-k * E)
        g = A - (A - A0) * e
        return g, np.column_stack([1 - e, e, (A - A0) * E * e * k])
    raise ValueError(kind)


def curve_prior(kind, th):
    """Negative log prior (penalty) and its gradient, preregistered priors."""
    if kind == "M0":
        return th[0] ** 2 / (2 * 1.5 ** 2), np.array([th[0] / 1.5 ** 2])
    if kind == "M1a":
        return (th[0] ** 2 / (2 * 1.5 ** 2) + th[1] ** 2 / 2,
                np.array([th[0] / 1.5 ** 2, th[1]]))
    A, A0, lk = th; m = np.log(0.1)
    return (A ** 2 / (2 * 1.5 ** 2) + A0 ** 2 / (2 * 1.5 ** 2) + (lk - m) ** 2 / 2,
            np.array([A / 1.5 ** 2, A0 / 1.5 ** 2, lk - m]))


NCURVE = {"M0": 1, "M1a": 2, "M1b": 3}
START = {"M0": [0.0], "M1a": [0.0, 0.0], "M1b": [0.5, -0.5, np.log(0.1)]}


# ---------------------------------------------------------------- data
def true_ability(h1, E):
    if h1 == "M0":
        return np.zeros(N)
    if h1 == "M1a":
        return LIN["lo"] + (LIN["hi"] - LIN["lo"]) * (E - E_RANGE[0]) / (E_RANGE[1] - E_RANGE[0])
    return ASYM["A"] - (ASYM["A"] - ASYM["A0"]) * np.exp(-ASYM["k"] * E)


def simulate(rng, k_solo, h1, h2, delta):
    """h2 in {'c','b','a'}: true pair model; delta added on the log-odds scale."""
    E = rng.uniform(*E_RANGE, N)
    g = true_ability(h1, E)
    v = rng.normal(0, M0_SD if h1 == "M0" else SIGMA_V, N)
    a = g + v
    perm = rng.permutation(N)                 # random assignment of people to schedule slots
    days = SCHED[k_solo]
    b = rng.normal(0, SNARE_SD, (len(days), N_SNARES))
    rows = []   # (i, j, snare_id, y, pair_id)
    pair_ids = {}
    for d, (pairs, solos) in enumerate(days):
        for x in solos:
            i = perm[x]
            p = expit(a[i] + b[d])
            y = rng.random(N_SNARES) < p
            for s in range(N_SNARES):
                rows.append((i, -1, d * N_SNARES + s, y[s], -1))
        for (x1, x2) in pairs:
            i, j = sorted((perm[x1], perm[x2]))
            pid = pair_ids.setdefault((i, j), len(pair_ids))
            if h2 == "c":
                u = 1 - (1 - expit(a[i] + b[d])) * (1 - expit(a[j] + b[d]))
                eta = np.log(u) - np.log1p(-u) + delta
            elif h2 == "b":
                eta = max(a[i], a[j]) + b[d] + delta
            else:  # pooled experience
                eta = true_ability(h1, np.array([E[i] + E[j]]))[0] + b[d] + delta
            y = rng.random(N_SNARES) < expit(eta)
            for s in range(N_SNARES):
                rows.append((i, j, d * N_SNARES + s, y[s], pid))
    return E, np.array(rows, dtype=int), len(days) * N_SNARES


# ---------------------------------------------------------------- H1 fit (solo only)
def h1_negll(theta, kind, P, S, Y, E, Em, Es, nb, persons):
    nc = NCURVE[kind]
    th, ls, b = theta[:nc], theta[nc], theta[nc + 1:]
    sig = np.exp(ls)
    g, J = curve(kind, th, E, Em, Es)
    eta = g[P][:, None] + sig * GH_Z[None, :] + b[S][:, None]
    llo = Y[:, None] * eta - np.logaddexp(0, eta)
    lq = np.zeros((N, len(GH_Z)))
    np.add.at(lq, P, llo)
    lq = lq[persons]
    Li = logsumexp(lq + GH_LW, axis=1)
    pi_full = np.zeros((N, len(GH_Z)))
    pi_full[persons] = np.exp(lq + GH_LW - Li[:, None])
    rw = (Y[:, None] - expit(eta)) * pi_full[P]
    rs = rw.sum(1)
    gg = np.bincount(P, rs, minlength=N)
    grad_b = np.bincount(S, rs, minlength=nb)
    grad_ls = sig * np.sum(rw * GH_Z[None, :])
    pen_c, gpen_c = curve_prior(kind, th)
    # sigma ~ Exponential(1), parameterised on log scale (with Jacobian)
    nll = -Li.sum() + pen_c + sig - ls + 0.5 * np.sum(b ** 2)
    grad = np.concatenate([-(J.T @ gg) + gpen_c, [-grad_ls + sig - 1], -grad_b + b])
    return nll, grad


def h1_fit(kind, data, E, nb, persons):
    Em, Es = E.mean(), E.std()
    m = (data[:, 1] < 0) & np.isin(data[:, 0], persons)
    P, S, Y = data[m, 0], data[m, 2], data[m, 3]
    x0 = np.concatenate([START[kind], [np.log(0.5)], np.zeros(nb)])
    res = minimize(h1_negll, x0, args=(kind, P, S, Y, E, Em, Es, nb, persons),
                   jac=True, method="L-BFGS-B")
    return res.x


def h1_person_ll(theta, kind, data, E, nb, person, Em, Es):
    nc = NCURVE[kind]
    th, sig, b = theta[:nc], np.exp(theta[nc]), theta[nc + 1:]
    g, _ = curve(kind, th, E, Em, Es)
    m = (data[:, 1] < 0) & (data[:, 0] == person)
    S, Y = data[m, 2], data[m, 3]
    eta = g[person] + sig * GH_Z[None, :] + b[S][:, None]
    lq = (Y[:, None] * eta - np.logaddexp(0, eta)).sum(0)
    return logsumexp(lq + GH_LW)


def choose(elpd_pointwise, names):
    """Preregistered rule: best model must beat the next best by > 2 SE of the difference."""
    tot = {k: v.sum() for k, v in elpd_pointwise.items()}
    order = sorted(names, key=lambda k: -tot[k])
    best, second = order[0], order[1]
    diff = elpd_pointwise[best] - elpd_pointwise[second]
    se = np.sqrt(len(diff) * diff.var(ddof=1))
    return best, (best if diff.sum() > 2 * se else "none")


def run_h1(args):
    seed, k_solo, h1 = args
    rng = np.random.default_rng(seed)
    E, data, nb = simulate(rng, k_solo, h1, "c", 0.0)
    Em, Es = E.mean(), E.std()
    out = {}
    elpd = {}
    for kind in ["M0", "M1a", "M1b"]:
        full = h1_fit(kind, data, E, nb, np.arange(N))
        out[kind] = full
        pts = []
        for i in range(N):
            keep = np.array([x for x in range(N) if x != i])
            th = h1_fit(kind, data, E, nb, keep)
            pts.append(h1_person_ll(th, kind, data, E, nb, i, Em, Es))
        elpd[kind] = np.array(pts)
    top, chosen = choose(elpd, ["M0", "M1a", "M1b"])
    # experience effect: linear slope in log-odds per 10 years (from M1a fit), and E90 from M1b
    slope10 = out["M1a"][1] / Es * 10
    e90 = np.log(10) / np.exp(out["M1b"][2])
    sigma_hat = np.exp(out["M1b" if h1 == "M1b" else ("M1a" if h1 == "M1a" else "M0")][NCURVE[h1]])
    return dict(k=k_solo, h1=h1, top=top, chosen=chosen, slope10=slope10, e90=e90, sigma=sigma_hat)


# ---------------------------------------------------------------- H2 joint fit
def h2_negll(theta, kind, curvekind, sig, I, J, S, Y, E, Em, Es, nb):
    nc = NCURVE[curvekind]
    th, v, b, d = theta[:nc], theta[nc:nc + N], theta[nc + N:nc + N + nb], theta[-1]
    g, Jg = curve(curvekind, th, E, Em, Es)
    a = g + v
    solo = J < 0
    JJ = np.where(solo, 0, J)
    xi = a[I] + b[S]
    gi = np.zeros(len(Y)); gj = np.zeros(len(Y)); gb = np.zeros(len(Y)); gpool = np.zeros(len(Y))
    if kind == "c":
        qi, qj = expit(xi), expit(a[JJ] + b[S])
        u = np.clip(1 - (1 - qi) * (1 - qj), 1e-12, 1 - 1e-12)
        eta_pair = np.log(u) - np.log1p(-u) + d
        gi_p, gj_p, gb_p = qi / u, qj / u, (qi + qj) / u
    elif kind == "b":
        ai, aj = a[I], a[JJ]
        top_i = ai >= aj
        eta_pair = np.where(top_i, ai, aj) + b[S] + d
        gi_p, gj_p, gb_p = top_i.astype(float), (~top_i).astype(float), np.ones(len(Y))
    else:
        Esum = E[I] + E[JJ]
        gp, Jp = curve(curvekind, th, Esum, Em, Es)
        eta_pair = gp + b[S] + d
        gi_p, gj_p, gb_p = np.zeros(len(Y)), np.zeros(len(Y)), np.ones(len(Y))
    eta = np.where(solo, xi, eta_pair)
    r = Y - expit(eta)
    ll = np.sum(Y * eta - np.logaddexp(0, eta))
    gi = r * np.where(solo, 1.0, gi_p)
    gj = r * np.where(solo, 0.0, gj_p)
    gb = r * np.where(solo, 1.0, gb_p)
    ga = np.bincount(I, gi, minlength=N) + np.bincount(JJ, gj, minlength=N)
    grad_th = Jg.T @ ga
    if kind == "a":
        grad_th = grad_th + Jp[~solo].T @ r[~solo]
    grad_v = ga
    grad_b = np.bincount(S, gb, minlength=nb)
    grad_d = np.sum(r * ~solo)
    pen_c, gpen_c = curve_prior(curvekind, th)
    nll = -ll + pen_c + np.sum(v ** 2) / (2 * sig ** 2) + 0.5 * np.sum(b ** 2) + d ** 2 / 2
    grad = np.concatenate([-grad_th + gpen_c, -grad_v + v / sig ** 2, -grad_b + b, [-grad_d + d]])
    return nll, grad


def h2_fit(kind, curvekind, sig, data, E, nb, mask):
    Em, Es = E.mean(), E.std()
    I, J, S, Y = data[mask, 0], data[mask, 1], data[mask, 2], data[mask, 3]
    x0 = np.concatenate([START[curvekind], np.zeros(N), np.zeros(nb), [0.0]])
    res = minimize(h2_negll, x0, args=(kind, curvekind, sig, I, J, S, Y, E, Em, Es, nb),
                   jac=True, method="L-BFGS-B")
    return res.x


def h2_pointll(theta, kind, curvekind, data, E, nb, mask):
    """Log-likelihood of each row in mask (pair rows), using point estimates."""
    Em, Es = E.mean(), E.std()
    nc = NCURVE[curvekind]
    th, v, b, d = theta[:nc], theta[nc:nc + N], theta[nc + N:nc + N + nb], theta[-1]
    g, _ = curve(curvekind, th, E, Em, Es)
    a = g + v
    I, J, S, Y = data[mask, 0], data[mask, 1], data[mask, 2], data[mask, 3]
    if kind == "c":
        u = np.clip(1 - (1 - expit(a[I] + b[S])) * (1 - expit(a[J] + b[S])), 1e-12, 1 - 1e-12)
        eta = np.log(u) - np.log1p(-u) + d
    elif kind == "b":
        eta = np.maximum(a[I], a[J]) + b[S] + d
    else:
        gp, _ = curve(curvekind, th, E[I] + E[J], Em, Es)
        eta = gp + b[S] + d
    return Y * eta - np.logaddexp(0, eta)


def run_h2(args):
    seed, k_solo, h1, h2, delta = args
    rng = np.random.default_rng(seed)
    E, data, nb = simulate(rng, k_solo, h1, h2, delta)
    # sigma from the H1 (solo-only) fit with the true curve form
    th1 = h1_fit(h1, data, E, nb, np.arange(N))
    sig = max(np.exp(th1[NCURVE[h1]]), 0.05)
    pid = data[:, 4]
    is_pair = data[:, 1] >= 0
    folds = np.random.default_rng(seed + 7).permutation(45) % 5
    elpd = {}
    dhat = {}
    for kind in ["a", "b", "c"]:
        full = h2_fit(kind, h1, sig, data, E, nb, np.ones(len(data), bool))
        dhat[kind] = full[-1]
        per_pair = np.zeros(45)
        for f in range(5):
            held = is_pair & (folds[np.where(is_pair, pid, 0)] == f)
            th = h2_fit(kind, h1, sig, data, E, nb, ~held)
            ll = h2_pointll(th, kind, h1, data, E, nb, held)
            np.add.at(per_pair, pid[held], ll)
        elpd[kind] = per_pair
    top, chosen = choose(elpd, ["a", "b", "c"])
    return dict(k=k_solo, h1=h1, h2=h2, delta=delta, top=top, chosen=chosen,
                d_c=dhat["c"], d_a=dhat["a"])


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    n1 = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    n2 = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    tasks1 = [(1000 * r + 13 * k + ["M0", "M1a", "M1b"].index(h), k, h) for r in range(n1)
              for k in (2, 3) for h in ("M0", "M1a", "M1b")]
    tasks1 = [(abs(s) % (2 ** 31), k, h) for s, k, h in tasks1]
    scen2 = [("M1b", "c", 0.0), ("M1b", "c", 0.5), ("M1b", "b", 0.0), ("M1b", "a", 0.0), ("M1b", "a", 0.5)]
    tasks2 = [(500000 + 1000 * r + 17 * k + i, k, *sc) for r in range(n2)
              for k in (2, 3) for i, sc in enumerate(scen2)]
    with Pool(workers) as pool:
        res1 = pool.map(run_h1, tasks1, chunksize=4)
        res2 = pool.map(run_h2, tasks2, chunksize=2)
    import json
    with open(sys.argv[4] if len(sys.argv) > 4 else "sim_results.json", "w") as fh:
        json.dump({"h1": res1, "h2": res2}, fh, default=float)
    print("done", len(res1), len(res2))
