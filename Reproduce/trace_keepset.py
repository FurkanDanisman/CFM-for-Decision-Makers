"""Compute the transitive set of files the reproduction entry points need.

Enumerating the keep-set by hand is how you end up with a branch that is
missing one import and fails three steps in. This walks out from the entry
points instead:

  *.py      -> `import X` / `from X import` resolved against repo dirs, plus
               any 'path/to/file.py' string literal
  *.sbatch  -> $REPO/... paths and bare *.py / *.sbatch / *.sh references
  *.sh      -> same

Prints KEEP (reachable) and DROP (everything else tracked by git), so the
prune is reviewable before anything is deleted.

    python Reproduce/trace_keepset.py            # summary
    python Reproduce/trace_keepset.py --list-keep
    python Reproduce/trace_keepset.py --list-drop
"""
from __future__ import annotations
import argparse, ast, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENTRY = [
    # --- Training -------------------------------------------------------
    "benchmarks/cluster/submit_train_cpfn2d_j32_random.sbatch",
    "benchmarks/cluster/submit_train_causalpfn_j1024_headrand.sbatch",
    "benchmarks/cluster/submit_train_graph2d.sbatch",
    "training_dopfn_base/train.py",
    # --- ComplexMech ----------------------------------------------------
    "UWYK_Fig3_4/generate_pehe_benchmark.py",
    "benchmarks/cluster/submit_cmech_pehe_1d_vs_2d.sbatch",
    "benchmarks/cluster/submit_cmech_density_score.sbatch",
    "benchmarks/aggregate_cmech_methods.py",
    "benchmarks/aggregate_cmech_1d_vs_2d.py",
    "UWYK_Fig3_4/plot_1d_vs_2d.py",
    "UWYK_Fig3_4/cate_ate_distributions.py",
    # --- RealCause ------------------------------------------------------
    "benchmarks/cluster/submit_realcause_density_unified.sbatch",
    "realcause_eval/summarize_realcause.py",
    # --- Case study -----------------------------------------------------
    "case_study/d_variation/generation_d.py",
    "benchmarks/cluster/submit_cs_dvar_density.sbatch",
    "case_study/density_eval/submit_score_cpfn_schema.sbatch",
    "case_study/density_eval/summarize_case_tables.py",
    "realcause_eval/aggregate_scm_ctx_sweep.py",
    "realcause_eval/plot_scm_boxplots.py",
    "realcause_eval/plot_scm_heatmap.py",
    # --- shared scorer + diagnostics ------------------------------------
    "UWYK_Fig3_4/cate_density_metrics.py",
    "benchmarks/all_bench_progress.py",
    "benchmarks/audit_point_vs_density.py",
    "benchmarks/compare_bench_schemas.py",
]

SEARCH_DIRS = ["", "benchmarks", "realcause_eval", "realcause_eval/Table1",
               "UWYK_Fig3_4", "case_study", "case_study/eval",
               "case_study/density_eval", "case_study/d_variation",
               "benchmarks/eval_graph2d", "benchmarks/eval_causalpfn2d",
               "benchmarks/eval_scm_case_studies", "benchmarks/l2_ihdp",
               "benchmarks/empirical_tests", "benchmarks/uwyk_table1/shims",
               "training", "training_dopfn_base", "training_causalpfn2d",
               "training_graph2d", "rpfn_patches", "dopfn_patches",
               "models", "losses"]

_REF = re.compile(r"[\w./$\{\}-]+\.(?:py|sbatch|sh)\b")


def resolve_module(mod: str) -> list[str]:
    out = []
    rel = mod.replace(".", os.sep)
    for d in SEARCH_DIRS:
        for cand in (os.path.join(d, rel + ".py"),
                     os.path.join(d, rel, "__init__.py")):
            if os.path.isfile(os.path.join(ROOT, cand)):
                out.append(os.path.normpath(cand))
    return out


def refs_from_py(path: str) -> set[str]:
    out: set[str] = set()
    try:
        src = open(os.path.join(ROOT, path), encoding="utf-8").read()
    except Exception:
        return out
    try:
        tree = ast.parse(src)
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    out.update(resolve_module(a.name))
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                out.update(resolve_module(n.module))
    except SyntaxError:
        pass
    for m in _REF.findall(src):
        out.update(resolve_ref(m, path))
    return out


def resolve_ref(tok: str, origin: str) -> list[str]:
    tok = re.sub(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?/?", "", tok).lstrip("/")
    if not tok or tok.startswith("."):
        return []
    cands = [tok, os.path.join(os.path.dirname(origin), os.path.basename(tok))]
    cands += [os.path.join(d, os.path.basename(tok)) for d in SEARCH_DIRS]
    seen = []
    for c in cands:
        c = os.path.normpath(c)
        if os.path.isfile(os.path.join(ROOT, c)) and c not in seen:
            seen.append(c)
    return seen[:1] if seen else []


def refs_from_script(path: str) -> set[str]:
    out: set[str] = set()
    try:
        src = open(os.path.join(ROOT, path), encoding="utf-8").read()
    except Exception:
        return out
    for m in _REF.findall(src):
        out.update(resolve_ref(m, path))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-keep", action="store_true")
    ap.add_argument("--list-drop", action="store_true")
    a = ap.parse_args()

    keep, queue, missing = set(), list(ENTRY), []
    for e in ENTRY:
        if not os.path.isfile(os.path.join(ROOT, e)):
            missing.append(e)
    while queue:
        cur = os.path.normpath(queue.pop())
        if cur in keep or not os.path.isfile(os.path.join(ROOT, cur)):
            continue
        keep.add(cur)
        nxt = refs_from_py(cur) if cur.endswith(".py") else refs_from_script(cur)
        queue.extend(n for n in nxt if n not in keep)

    # A package whose modules are kept needs its __init__.py, and so does every
    # parent package up the chain -- otherwise the import that put the module
    # in the keep-set fails on the pruned branch.
    for f in list(keep):
        d = os.path.dirname(f)
        while d:
            ini = os.path.join(d, "__init__.py")
            if os.path.isfile(os.path.join(ROOT, ini)):
                keep.add(os.path.normpath(ini))
            d = os.path.dirname(d)

    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                             text=True).stdout.split()
    code = [f for f in tracked if f.endswith((".py", ".sbatch", ".sh"))]
    drop = sorted(set(code) - keep)

    if missing:
        print("MISSING ENTRY POINTS:")
        for m in missing:
            print("  ", m)
        print()
    print(f"entry points : {len(ENTRY)}")
    print(f"KEEP (code)  : {len(keep)}")
    print(f"DROP (code)  : {len(drop)}  of {len(code)} tracked code files")
    if a.list_keep:
        print("\n--- KEEP ---")
        for f in sorted(keep):
            print(f)
    if a.list_drop:
        print("\n--- DROP ---")
        for f in drop:
            print(f)


if __name__ == "__main__":
    main()
