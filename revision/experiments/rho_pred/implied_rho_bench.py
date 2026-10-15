"""Predicted (implied) cross-world correlation of the 2D heads on the benchmarks, from their dumped joints.
python implied_rho_bench.py OUT_PREFIX SPEC [SPEC ...]
    SPEC = "model|benchmark|root|tag|must"   (tag = '' for the primary joint, or e.g. v3a for p_joint_scaled_v3a;
                                             must = a path component every file must contain, '' for none)

Every *r###.npz under root (recursively) that holds a joint is read; per query rho_pred is computed exactly as
for the in-prior figure (pred_rho_shard.rho_from_joint: correlation of the predicted joint on the interior grid,
cells uniform and independent); per dataset the median over its queries. The setting (dataset, case study, graph
size or rho bin) is taken from the path. For the controlled-rho dumps (benchmark rho_ctrl_*), the dataset's TRUE rho
(rho_true stored by the generators) is paired with it, so predicted and true coupling can be compared per bin.
Output: OUT_PREFIX.csv (one row per dataset) and OUT_PREFIX.md (mean +- sd per model, benchmark, setting).
"""
import csv
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pred_rho_shard import rho_from_joint                      # noqa: E402

SETTINGS = ["PSID_bal", "PSIDbal", "IHDP", "ACIC", "CPS", "PSID", "Observed_Mediator_and_Confounder", "Observed_Confounder",
            "Observed_Mediator", "Unobserved_Confounder", "Frontdoor_Criterion", "Backdoor_Criterion"]


def setting_of(path, bench):
    parts = path.split(os.sep)
    if bench.startswith("rho_ctrl"):
        b = next((p for p in parts if re.fullmatch(r"bin[0-2]", p)), "?")
        c = next((s for s in SETTINGS if s in parts), "ComplexMech")
        return f"{c} {b}"
    for s in SETTINGS:
        if s in parts or any(p.startswith(s + "_") or p == "SEMIREAL_" + s for p in parts):
            return "PSID_bal" if s == "PSIDbal" else s
    m = re.search(r"CMECH_n(\d+)", path)
    if m:
        return f"n={m.group(1)}"
    m = re.search(r"/(\d+)node/", path)
    return f"n={m.group(1)}" if m else "?"


def per_file(f, tag):
    z = np.load(f, allow_pickle=True)
    key = f"p_joint_scaled_{tag}" if tag else "p_joint_scaled"
    if key not in z.files:
        return None
    P = np.asarray(z[key], dtype=np.float64)
    if P.ndim != 3:
        return None
    K = P.shape[-1]
    e = np.asarray(z["edges"], dtype=np.float64).reshape(-1) if "edges" in z.files else None
    if e is not None and e.size == K + 1:
        w = float(np.mean(np.diff(e))); c = 0.5 * (e[:-1] + e[1:])
    else:
        w = 2.0 / K; c = -1.0 + w * (np.arange(K) + 0.5)
    r = rho_from_joint(P, c, w)
    r = r[np.isfinite(r)]
    return float(np.median(r)) if r.size else None


def true_rho(f, bench, root):
    """rho_true of the source dataset of a controlled-rho dump (dump index -> source file)."""
    m = re.search(r"r(\d+)\.npz$", f)
    if not m:
        return None
    k = int(m.group(1)); D = os.path.join(os.environ.get("SCRATCH", ""), "rho_ctrl")
    parts = f.split(os.sep)
    b = next((p for p in parts if re.fullmatch(r"bin[0-2]", p)), None)
    if b is None:
        return None
    case = next((s for s in SETTINGS if s in parts), None)
    src = (os.path.join(D, "cs", b, case, "N1000", f"{case}_{k}.npz") if case else
           os.path.join(D, "cm", b, "complexmech", "5node", "path_TY", "hide_0.0", f"r{k}.npz"))
    if not os.path.isfile(src):
        return None
    return float(np.load(src, allow_pickle=True)["rho_true"])


def main(out, specs):
    rows = []
    for spec in specs:
        model, bench, root, tag, must = (spec.split("|") + ["", ""])[:5]
        files = [f for f in glob.glob(os.path.join(root, "**", "*r[0-9][0-9]*.npz"), recursive=True)
                 if "summary" not in os.path.basename(f) and (not must or must in f.split(os.sep))]
        n_ok = 0
        for f in sorted(files):
            try:
                r = per_file(f, tag)
            except Exception as exc:                            # recorded, never silently dropped
                print(f"[{model} {bench}] cannot read {f}: {exc}", flush=True)
                continue
            if r is None:
                continue
            row = dict(model=model, benchmark=bench, setting=setting_of(f, bench), rho_pred=r,
                       rho_true=true_rho(f, bench, root) if bench.startswith("rho_ctrl") else "", file=f)
            rows.append(row); n_ok += 1
        print(f"{model:16s} {bench:10s} {root}: {len(files)} files, {n_ok} with a joint", flush=True)
    if not rows:
        sys.exit("no joints found under any root")
    with open(out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    md = ["# Implied (predicted) ρ of the 2D heads on the benchmarks", "",
          "Per dataset: median over its queries of the correlation of the head's predicted joint. Mean ± sd over datasets "
          "(n). For the controlled-ρ benchmarks, the true ρ of the same datasets is shown next to it.", "",
          "| model | benchmark | setting | n | predicted ρ (mean ± sd) | true ρ (mean ± sd) |", "|---|---|---|---|---|---|"]
    keys = sorted({(r["model"], r["benchmark"], r["setting"]) for r in rows})
    for k in keys:
        v = np.array([r["rho_pred"] for r in rows if (r["model"], r["benchmark"], r["setting"]) == k])
        t = np.array([r["rho_true"] for r in rows if (r["model"], r["benchmark"], r["setting"]) == k
                      and r["rho_true"] not in ("", None)], dtype=float)
        tt = f"{t.mean():+.3f} ± {t.std(ddof=1):.3f}" if t.size > 1 else "—"
        md.append(f"| {k[0]} | {k[1]} | {k[2]} | {v.size} | {v.mean():+.3f} ± {v.std(ddof=1) if v.size > 1 else 0:.3f} | {tt} |")
    open(out + ".md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
