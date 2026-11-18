import json, sys
import numpy as np
R = json.load(open(sys.argv[1]))
h1 = R["h1"]; h2 = R["h2"]
print("=== H1: experience model recovery (solo trials only) ===")
for truth in ["M0", "M1a", "M1b"]:
    for k in (2, 3):
        rs = [r for r in h1 if r["h1"] == truth and r["k"] == k]
        n = len(rs)
        top = np.mean([r["top"] == truth for r in rs])
        ok = np.mean([r["chosen"] == truth for r in rs])
        wrong = np.mean([r["chosen"] not in (truth, "none") for r in rs])
        none = np.mean([r["chosen"] == "none" for r in rs])
        extra = ""
        if truth == "M1a":
            s = np.array([r["slope10"] for r in rs]); extra = f"slope/10y est mean {s.mean():.2f} (true 0.57) SD {s.std():.2f}"
        if truth == "M1b":
            e = np.array([r["e90"] for r in rs]); extra = f"E90 median {np.median(e):.1f} (true 15.4), 10-90% {np.percentile(e,10):.1f}-{np.percentile(e,90):.1f}"
        print(f"truth {truth} k={k} n={n}: best ELPD {top:.0%} | rule picks truth {ok:.0%}, wrong {wrong:.0%}, undecided {none:.0%}  {extra}")
print()
print("=== H2: pair benchmark recovery ===")
name = {"a": "pooled", "b": "best member", "c": "independent"}
for (h, d) in [("c", 0.0), ("c", 0.5), ("b", 0.0), ("a", 0.0), ("a", 0.5)]:
    for k in (2, 3):
        rs = [r for r in h2 if r["h2"] == h and r["delta"] == d and r["k"] == k]
        top = np.mean([r["top"] == h for r in rs])
        ok = np.mean([r["chosen"] == h for r in rs])
        wrong = np.mean([r["chosen"] not in (h, "none") for r in rs])
        none = np.mean([r["chosen"] == "none" for r in rs])
        print(f"truth {name[h]:12s} delta={d} k={k} n={len(rs)}: best ELPD {top:.0%} | rule picks truth {ok:.0%}, wrong {wrong:.0%}, undecided {none:.0%}")
print()
print("=== delta precision ===")
for col, h in [("d_c", "c"), ("d_a", "a")]:
    for k in (2, 3):
        null = np.array([r[col] for r in h2 if r["h2"] == h and r["delta"] == 0.0 and r["k"] == k])
        alt = np.array([r[col] for r in h2 if r["h2"] == h and r["delta"] == 0.5 and r["k"] == k])
        se = null.std()
        power_raw = np.mean(np.abs(alt) > 1.96 * se)
        power_corr = np.mean(np.abs(alt - null.mean()) > 1.96 * se)
        fp = np.mean(np.abs(null) > 1.96 * se)
        print(f"{col} k={k}: null mean {null.mean():+.2f}, SE {se:.2f}; delta=0.5 mean {alt.mean():.2f}; "
              f"detect 0.5: {power_raw:.0%} (bias-corrected {power_corr:.0%}); false positives at 0: {fp:.0%}")
