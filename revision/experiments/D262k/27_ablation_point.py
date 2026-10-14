#!/usr/bin/env python
"""Stage 27: point estimates (sqrt PEHE, relative eps_ATE) of the matched-label and matched-resolution
ablation models and their 2D counterparts, on the benchmarks of the ablation calibration tables
(tab:rc-ablation-cal, tab:cs-abl-cal, tab:cm-abl-r99-cal). Job body: 27_ablation_point.sh.

Scores existing dumps only, with the scripts the paper's main point tables use:
  RealCause    realcause_eval/point_raw_em.py --modes raw --ate-metric rel   (as 17_collect_t1.sh)
  case study   benchmarks/cs_point_total.py --roots <final_table ROOTS labels> (pooled over shifts, d=5..50)
  ComplexMech  benchmarks/cmech_point_total.py --data $SCRATCH/cmech_data_v2 (all rho, N=1000; as 17_collect_t1.sh),
               over a folder of symlinks to the model dump dirs (the script scores every dir under --dumps)
Delta (%) = 100 (1 - geometric mean over columns of 2D / ablation), per metric; + favours 2D (update_tables._delta).

    python 27_ablation_point.py --repo $REPO --scratch $SCRATCH --work $SCRATCH/ablation_point --out-dir <dir>
"""
import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

RC = ["IHDP", "ACIC", "CPS", "PSID", "PSID_bal"]
CASES = ["Observed_Confounder", "Observed_Mediator", "Observed_Mediator_and_Confounder",
         "Unobserved_Confounder", "Frontdoor_Criterion", "Backdoor_Criterion"]
CASE_TEX = ["Obs. Conf.", "Obs. Med.", "Obs. M+C", "Unobs. Conf.", "Frontdoor", "Backdoor"]
NS = [5, 10, 20, 30, 40, 50]
VARIANTS = ["matched label", "matched resolution", "2D"]

# family, variant,
#   RealCause: dump root (rel. to $SCRATCH), point_raw_em method row
#   case study: final_table ROOTS label, cs_point_total model name
#   ComplexMech: dump dir (rel. to $SCRATCH), symlink name (= cmech_point_total model dir), model name
S = [
    ("Do-PFN", "matched label",
     "dumps_all/dopfn_1d_botharms_262k_t1/rc", "dopfn_native",
     "dopfn_1d_botharms_262k_t1", "dopfn_1d_botharms_262k_t1",
     "cmech_dumps/dopfn_1d_botharms_262k_t1", "dopfn_1d_botharms_262k_t1", "dopfn_1d_botharms_262k_t1"),
    ("Do-PFN", "matched resolution",
     "dumps_all/dopfn_repro_1d_J10_262k_t1/rc", "dopfn_native",
     "dopfn_repro_1d_J10_262k_t1", "dopfn_repro_1d_J10_262k_t1",
     "cmech_dumps/dopfn_repro_1d_J10_262k_t1", "dopfn_repro_1d_J10_262k_t1", "dopfn_repro_1d_J10_262k_t1"),
    ("Do-PFN", "2D",
     "dumps_all/dopfn_repro_joint2d_262k_t1/rc", "dopfn_native",
     "joint2d_262k_t1", "dopfn_repro_joint2d_262k_t1",
     "cmech_dumps/dopfn_repro_joint2d_262k_t1", "dopfn_repro_joint2d_262k_t1", "dopfn_repro_joint2d_262k_t1"),
]
for fam, tag in (("UWYK No-Anc", "noanc"), ("UWYK Anc", "v3a")):
    S += [
        (fam, "matched label",
         "aurora3_kill/rc/uwyk_C_botharms", f"uwyk1d-{tag}",
         "uwyk_C_botharms", f"uwyk_C_botharms-{tag}",
         "aurora3_allrho/cm_rho99/uwyk_C_botharms", "uwyk_C_botharms", f"uwyk_C_botharms-{tag}"),
        (fam, "matched resolution",
         "aurora3_kill/rc/uwyk_D_j32_nobin", f"uwyk1d-{tag}",
         "uwyk_D_j32_nobin", f"uwyk_D_j32_nobin-{tag}",
         "aurora3_allrho/cm_rho99/uwyk_D_j32_nobin", "uwyk_D_j32_nobin", f"uwyk_D_j32_nobin-{tag}"),
        (fam, "2D",
         "rc_dens_uni", f"graph2d-{tag}",
         "orig", f"graph2d-{tag}",
         "cmech_dumps/graph2d", "graph2d", f"graph2d-{tag}"),
    ]
S += [
    ("CausalPFN-C", "matched label",
     "dumps_all/cpfn1d_botharms/rc", "cpfn1d",
     "botharms", "cpfn1d_botharms",
     "cmech_dumps/cpfn1d_botharms", "cpfn1d_botharms", "cpfn1d_botharms"),
    ("CausalPFN-C", "matched resolution",
     "dumps_all/cpfn1d_j32/rc", "cpfn1d",
     "j32", "cpfn1d_j32",
     "cmech_dumps/cpfn1d_j32", "cpfn1d_j32", "cpfn1d_j32"),
    ("CausalPFN-C", "2D",
     "rc_dens_eta0", "cpfn2d",
     "eta0", "cpfn2d_eta0",
     "cmech_dumps/cpfn2d_eta0", "cpfn2d_eta0", "cpfn2d_eta0"),
]
FAMILIES = ["Do-PFN", "UWYK No-Anc", "UWYK Anc", "CausalPFN-C"]


def run(cmd, log):
    print("[run]", " ".join(cmd), ">", log, flush=True)
    with open(log, "w") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode:
        print(f"  FAILED (exit {r.returncode}): {log}", flush=True)
    return r.returncode


def parse_point_md(path, method):
    """(n, pehe, pehe_se, eps, eps_se) of one method row of a point_raw_em table (--modes raw), or None."""
    if not os.path.exists(path):
        return None
    for l in open(path):
        c = [x.strip() for x in l.strip().strip("|").split("|")]
        if len(c) >= 4 and c[0] == method and "±" in c[2]:
            pe, ps = (float(x) for x in c[2].split("±"))
            ea, es = (float(x) for x in c[3].split("±"))
            return int(c[1]), pe, ps, ea, es
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--scratch", default=os.environ.get("SCRATCH", ""))
    ap.add_argument("--work", required=True, help="scratch folder for the raw script outputs")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cs-workers", type=int, default=10)
    a = ap.parse_args()
    R, SC, W = a.repo, a.scratch, a.work
    logs = os.path.join(W, "logs"); os.makedirs(logs, exist_ok=True)
    py = sys.executable

    # ---- inventory: report every root up front; missing ones become '—' rows, never a silent zero
    for fam, var, rc, rcm, csl, csm, cmd, cml, cmm in S:
        for kind, p in (("rc", rc), ("cm", cmd)):
            q = os.path.join(SC, p)
            print(f"[root] {fam:12s} {var:18s} {kind} {'ok     ' if os.path.isdir(q) else 'MISSING'} {q}", flush=True)
    sys.path.insert(0, os.path.join(R, "benchmarks"))
    from final_table import ROOTS                                    # noqa: E402
    roots = {lab: cs for lab, _rc, cs, _single in ROOTS}
    for lab in sorted({s[4] for s in S}):
        q = os.path.join(SC, roots[lab])
        cell = os.path.join(q, "shift0", "d5", "ctx1000")
        sub = sorted(os.listdir(cell)) if os.path.isdir(cell) else "MISSING"
        print(f"[root] cs {lab:28s} {'ok     ' if os.path.isdir(q) else 'MISSING'} {q}  shift0/d5 method dirs: {sub}",
              flush=True)

    # ---- RealCause: point_raw_em per (root, method, dataset)
    pt = os.path.join(R, "realcause_eval", "point_raw_em.py")
    rc_jobs = []
    for fam, var, rc, rcm, *_ in S:
        root = os.path.join(SC, rc)
        if not os.path.isdir(root):
            continue
        for ds in RC:
            md = os.path.join(W, "rc", rc.replace("/", "__"), f"point_raw_em_{ds}__{rcm}.md")
            os.makedirs(os.path.dirname(md), exist_ok=True)
            rc_jobs.append([py, pt, "--root", root, "--dataset", ds, "--methods", rcm,
                            "--modes", "raw", "--ate-metric", "rel", "--out-md", md])
    # ---- ComplexMech: symlink farm, one cmech_point_total pass
    farm = os.path.join(W, "cm_links"); os.makedirs(farm, exist_ok=True)
    for *_, cmd, cml, _cmm in S:
        tgt, ln = os.path.join(SC, cmd), os.path.join(farm, cml)
        if os.path.islink(ln):
            os.unlink(ln)
        if os.path.isdir(os.path.join(tgt, "N1000")):
            os.symlink(tgt, ln)
        else:
            print(f"[cm] MISSING {tgt}/N1000", flush=True)
    cm_csv = os.path.join(W, "cm_point_allrho.csv")
    cm_cmd = [py, "-u", os.path.join(R, "benchmarks", "cmech_point_total.py"), "--dumps", farm,
              "--data", os.path.join(SC, "cmech_data_v2"), "--nodes", *map(str, NS), "--csv", cm_csv,
              "--out", os.path.join(W, "cm_point_allrho.md")]
    # ---- case studies: one cs_point_total pass over the needed ROOTS labels
    cs_csv = os.path.join(W, "cs_point.csv")
    cs_cmd = [py, "-u", os.path.join(R, "benchmarks", "cs_point_total.py"), "--scratch", SC,
              "--roots", *sorted({s[4] for s in S}), "--workers", str(a.cs_workers),
              "--csv", cs_csv, "--out", os.path.join(W, "cs_point.md")]

    # no stale outputs from an earlier run can be collected if a script fails this time
    for f in [cs_csv, cm_csv] + [c[-1] for c in rc_jobs]:
        if os.path.exists(f):
            os.remove(f)
    with ThreadPoolExecutor(max_workers=6) as ex:
        fcs = ex.submit(run, cs_cmd, os.path.join(logs, "cs_point_total.log"))
        fcm = ex.submit(run, cm_cmd, os.path.join(logs, "cmech_point_total.log"))
        frc = [ex.submit(run, c, os.path.join(logs, "rc_" + os.path.basename(c[-1])[:-3] + "_"
                                              + c[c.index("--root") + 1].replace("/", "_")[-40:] + ".log"))
               for c in rc_jobs]
        rcs = [f.result() for f in frc]; ccs, ccm = fcs.result(), fcm.result()
    print(f"[done] rc failures {sum(1 for r in rcs if r)}/{len(rcs)}, cs exit {ccs}, cm exit {ccm}", flush=True)
    warn = [l.rstrip() for l in open(os.path.join(logs, "cmech_point_total.log")) if "WARN" in l]
    print(f"[cm] {len(warn)} dump-count WARN line(s)" + "".join("\n   " + w for w in warn), flush=True)

    # ---- collect: (bench, setting, family, variant) -> (pehe, pehe_se, eps, eps_se)
    res = {}
    for fam, var, rc, rcm, *_ in S:
        for ds in RC:
            r = parse_point_md(os.path.join(W, "rc", rc.replace("/", "__"), f"point_raw_em_{ds}__{rcm}.md"), rcm)
            if r:
                res[("RealCause", ds, fam, var)] = r[1:]
    if os.path.exists(cs_csv):
        name = {s[5]: (s[0], s[1]) for s in S}
        for row in csv.DictReader(open(cs_csv)):
            if row["model"] in name and row["d"] == "all":
                fam, var = name[row["model"]]
                res[("CaseStudy", row["case"], fam, var)] = tuple(
                    float(row[k]) for k in ("pehe", "pehe_sem", "eps_ate", "eps_sem"))
    if os.path.exists(cm_csv):
        name = {s[8]: (s[0], s[1]) for s in S}
        for row in csv.DictReader(open(cm_csv)):
            if row["model"] in name and int(row["context"]) == 1000:
                fam, var = name[row["model"]]
                res[("ComplexMech", str(int(row["nodes"])), fam, var)] = tuple(
                    float(row[k]) for k in ("pehe", "pehe_sem", "eps_ate", "eps_sem"))

    os.makedirs(a.out_dir, exist_ok=True)
    benches = [("RealCause", RC, RC, "RealCause (eps_ATE relative; point_raw_em.py --modes raw --ate-metric rel)"),
               ("CaseStudy", CASES, CASE_TEX, "Case studies (pooled over shifts 0/+2/-2 and d = 5..50; cs_point_total.py)"),
               ("ComplexMech", [str(n) for n in NS], [f"n={n}" for n in NS],
                "ComplexMech, all rho, N = 1000 (cmech_point_total.py, total = nonzero + zero queries)")]
    with open(os.path.join(a.out_dir, "ablation_point.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["benchmark", "setting", "model", "variant", "pehe", "pehe_se", "eps_ate", "eps_se"])
        for b, cols, _t, _h in benches:
            for fam in FAMILIES:
                for var in VARIANTS:
                    for c in cols:
                        v = res.get((b, c, fam, var))
                        if v:
                            w.writerow([b, c, fam, var] + [f"{x:.6f}" for x in v])

    def pm(v, se):
        return f"{v:.0f} ± {se:.0f}" if abs(v) >= 1000 else f"{v:.3f} ± {se:.3f}"

    def delta(fam, var, b, cols):
        """100 (1 - geomean_c(2D / ablation)) for PEHE and eps; needs every column of both rows"""
        a_ = [res.get((b, c, fam, var)) for c in cols]
        t_ = [res.get((b, c, fam, "2D")) for c in cols]
        if not all(a_) or not all(t_):
            return ["—", "—"]
        out = []
        for j in (0, 2):
            r = np.array([t[j] / x[j] for t, x in zip(t_, a_)])
            out.append(f"{100 * (1 - np.exp(np.mean(np.log(r)))):+.1f}")
        return out

    L = ["# Ablation point estimates (sqrt PEHE, eps_ATE)", "",
         "Mean ± SE over realizations, mode = raw. eps_ATE = |ATE_hat - ATE| / max(|ATE|, 0.1).",
         "Δ (%) on each ablation row = 100 (1 - geometric mean over the columns of 2D / ablation), per metric,",
         "from the unrounded means; positive favours the 2D model of the block. '—' = missing dumps.",
         "Do-PFN rows are the 262k temperature-off models (D262k stages 15/23).", ""]
    for b, cols, heads, title in benches:
        L += [f"## {title}", "",
              "| Model | " + " | ".join(f"{h} √PEHE | {h} ε_ATE" for h in heads) + " | Δ √PEHE (%) | Δ ε_ATE (%) |",
              "|---|" + "---|" * (2 * len(cols) + 2)]
        for fam in FAMILIES:
            for var in VARIANTS:
                lab = f"{fam} 2D" if var == "2D" else f"{fam} ({var})"
                cells = []
                for c in cols:
                    v = res.get((b, c, fam, var))
                    cells += [pm(v[0], v[1]), pm(v[2], v[3])] if v else ["—", "—"]
                cells += ["--", "--"] if var == "2D" else delta(fam, var, b, cols)
                L.append(f"| {lab} | " + " | ".join(cells) + " |")
        L.append("")
    open(os.path.join(a.out_dir, "ablation_point.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    n_exp = sum(len(c) for _b, c, _t, _h in benches) * len(S)
    print(f"[out] {len(res)}/{n_exp} (benchmark, setting, model) cells filled -> {a.out_dir}/ablation_point.{{csv,md}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
