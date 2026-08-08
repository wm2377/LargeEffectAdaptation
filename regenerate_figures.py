"""Regenerate every manuscript figure as a publication-format raster (TIFF).

Snakemake is deliberately not used here (it would want to rebuild simulations); instead
this driver reproduces, for each figure, exactly what the corresponding `rule plot_*` in
the Snakefile passes to the figure script:

  * scripts with a `__main__` block are run through runpy with the argv that block expects;
  * scripts that only run under Snakemake (`if "snakemake" in globals()`) are given a shim
    `snakemake` object carrying the same input / params / output as the rule.

The rule-level constants (SDISTS, N2U_SWEEP, fnum, ...) are not re-typed: the Snakefile's
plain-Python header (everything above `wildcard_constraints:`) is exec'd and read directly,
so the two cannot drift.

Every savefig is forced to DPI dots-per-inch and, for TIFF, LZW compression, via a patch on
Figure.savefig -- the figure scripts hardcode their own dpi (150-400) and several save
through common_functions.make_figure_set_width.

Usage:
    python regenerate_figures.py [--ext .tif] [--dpi 600] [--outdir results/plots_submission]
                                 [--only Figure_1,Figure_S7] [--list]
"""

import argparse
import itertools
import os
import runpy
import sys
import traceback

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
SNAKEFILE = os.path.join(BASE, "Snakefile")


# --------------------------------------------------------------------------------------
# Snakefile header
# --------------------------------------------------------------------------------------

def load_snakefile_header():
    """Exec the Snakefile's plain-Python preamble and return its namespace.

    Everything above `wildcard_constraints:` is ordinary Python (imports, SDISTS, fnum,
    N2U_SWEEP, the FIGURE_* constants); the Snakemake-specific directives below it are not,
    so the file is cut there.
    """
    with open(SNAKEFILE) as fh:
        lines = fh.readlines()
    cut = next(i for i, ln in enumerate(lines) if ln.startswith("wildcard_constraints:"))
    ns = {"__name__": "snakefile_header", "__file__": SNAKEFILE}
    exec(compile("".join(lines[:cut]), SNAKEFILE, "exec"), ns)
    return ns


def expand(pattern, **fields):
    """Snakemake's expand(): cartesian product over the keyword lists.

    Formatting goes through str.format exactly as Snakemake's does, so the numeric tokens
    in the file names match the ones the simulation rules wrote.
    """
    keys = list(fields)
    values = [fields[k] if isinstance(fields[k], (list, tuple)) else [fields[k]]
              for k in keys]
    return [pattern.format(**dict(zip(keys, combo)))
            for combo in itertools.product(*values)]


# --------------------------------------------------------------------------------------
# snakemake shim
# --------------------------------------------------------------------------------------

class IOList(list):
    """Snakemake's input/output object: a positional list that also has named attributes."""

    def __init__(self, **named):
        flat = []
        for key, value in named.items():
            setattr(self, key, value)
            flat.extend(value if isinstance(value, (list, tuple)) else [value])
        super().__init__(flat)


class Params:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getitem__(self, key):
        return self.__dict__[key]


class SnakemakeShim:
    def __init__(self, input=None, output=None, params=None, wildcards=None):
        self.input = input if input is not None else IOList()
        self.output = output if output is not None else IOList()
        self.params = params if params is not None else Params()
        self.wildcards = wildcards if wildcards is not None else Params()
        self.config = {}
        self.threads = 1
        self.log = []
        self.resources = Params()


# --------------------------------------------------------------------------------------
# output-format patch
# --------------------------------------------------------------------------------------

def _flatten_tiff(path, dpi):
    """Composite the saved RGBA TIFF onto white and rewrite it as RGB.

    Matplotlib writes an RGBA buffer; most journals reject an alpha channel in submitted
    TIFFs. Compositing (rather than dropping the channel) preserves every semi-transparent
    element exactly as it renders on the white page -- which is why TIFF is used here at
    all: the EPS backend would flatten those elements to opaque instead.
    """
    from PIL import Image

    with Image.open(path) as im:
        if im.mode == "RGB":
            return
        rgba = im.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.split()[-1])
    flat.save(path, compression="tiff_lzw", dpi=(dpi, dpi))


def patch_savefig(dpi, ext):
    """Force every figure to `dpi`, and LZW-compress TIFF output.

    The figure scripts pass their own dpi= to savefig (and to make_figure_set_width), so
    overriding at the Figure.savefig level is the only single place that catches them all.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    original = Figure.savefig

    def savefig(self, fname, *args, **kwargs):
        kwargs["dpi"] = dpi
        name = fname if isinstance(fname, str) else ""
        is_tiff = name.lower().endswith((".tif", ".tiff"))
        if is_tiff:
            pil = dict(kwargs.get("pil_kwargs") or {})
            pil.setdefault("compression", "tiff_lzw")
            kwargs["pil_kwargs"] = pil
        result = original(self, fname, *args, **kwargs)
        if is_tiff:
            _flatten_tiff(name, dpi)
        return result

    Figure.savefig = savefig


# --------------------------------------------------------------------------------------
# figure targets
# --------------------------------------------------------------------------------------

def build_targets(ns, out_dir, ext):
    """One entry per output figure, mirroring the Snakefile's plot_* rules.

    Each target is a dict with:
        name    -- stem of the output file (also the --only selector)
        script  -- figure script to run
        kind    -- "argv" (script has a __main__) or "snakemake" (needs the shim)
        argv    -- extra sys.argv entries, for kind == "argv"
        make    -- callable returning a SnakemakeShim, for kind == "snakemake"
        outputs -- every file the target writes (for existence reporting)
    """
    sim = os.path.join(RESULTS, "simulations", "sdist{sdist}_N2U{N2U}_shift{shift}_sigma2{sigma2}.pkl")
    sim_s7 = os.path.join(RESULTS, "simulations_figure_S7", "sdist{sdist}_N2U{N2U}_shift{shift}_sigma2{sigma2}.pkl")
    analytic = os.path.join(RESULTS, "analytic_curves", "sdist{sdist}_shift{shift}_sigma2{sigma2}.pkl")
    analytic_fixed = os.path.join(RESULTS, "analytic_curves",
                                  "sdist{sdist}_shift{shift}_sigma2{sigma2}__with_fixed_effect_size_distribution.pkl")
    analytic_window = os.path.join(RESULTS, "analytic_curves",
                                   "sdist{sdist}_shift{shift}_sigma2{sigma2}__by_effect_size_window.pkl")
    fixprob = os.path.join(RESULTS, "fixation_probability", "S{S}_shift{shift}_sigma2{sigma2}.pkl")
    boundaries = os.path.join(RESULTS, "trajectory_class_boundaries", "S{S}_sigma2{sigma2}.pkl")
    phase_space = os.path.join(RESULTS, "phase_space", "paramset{paramset}.pkl")

    N2U_SWEEP = ns["N2U_SWEEP"]
    SDISTS = ns["SDISTS"]
    fnum = ns["fnum"]
    np = ns["np"]

    # the three (shift, sigma2) panels shared by Figures S6 / S8 / S9 / S10 / S11
    COMBOS = [(80, 40), (80, 80), (50, 80)]
    SHIFT_101 = list(np.linspace(0, 100, 101))
    SHIFT_42 = list(np.linspace(0, 100, 42))

    def out(name):
        return os.path.join(out_dir, name + ext)

    targets = []

    def argv_target(name, script, argv):
        targets.append(dict(name=name, script=script, kind="argv", argv=argv,
                            outputs=[a for a in argv if a.startswith(out_dir)]))

    def snake_target(name, script, make, outputs):
        targets.append(dict(name=name, script=script, kind="snakemake", make=make,
                            outputs=outputs))

    # -- Figures 1, 2, 3: no data inputs beyond what the scripts discover themselves -----
    argv_target("Figure_1", "Figure_1.py", [out("Figure_1")])
    argv_target("Figure_2", "Figure_2.py", [out("Figure_2")])
    argv_target("Figure_3", "Figure_3.py", [out("Figure_3")])

    # -- Figure 4: one per mutational input (main text + alternative) --------------------
    for n2u in [ns["FIGURE_MAIN_N2U"], ns["FIGURE_ALT_N2U"]]:
        sdist, shift, sigma2 = "expon", "80", "40"
        name = f"Figure_4__{sdist}_N2U{n2u}_shift{shift}_sigma2{sigma2}"
        snake_target(
            name, "Figure_4.py",
            (lambda sdist=sdist, shift=shift, sigma2=sigma2, n2u=n2u, name=name: SnakemakeShim(
                input=IOList(
                    simulations_A=expand(sim, sdist=sdist, N2U=N2U_SWEEP, shift=shift, sigma2=sigma2),
                    analytic=expand(analytic, sdist=sdist, shift=SHIFT_101, sigma2=sigma2),
                ),
                output=IOList(figure_4=out(name)),
                params=Params(sdist=SDISTS[sdist], shift=float(shift),
                              sigma2=float(sigma2), N2U=float(n2u)),
            )),
            [out(name)])

    # -- Figure 5: full-output replicates at the manuscript parameters --------------------
    argv_target("Figure_5", "Figure_5.py",
                [os.path.join(RESULTS, "simulations_full_output",
                              "sdistexpon_N2U10_shift80_sigma240.pkl"),
                 out("Figure_5")])

    # -- Figure 6 and its phase-space variants (one script, four (paramset, mode, layout)) -
    for name, paramset, mode, smoothed, layout in [
        ("Figure_6", "default", "fixations", "smoothed", "standard"),
        ("Figure_S13", "default", "fixations", "raw", "standard"),
        ("Figure_S16", "default", "adaptation", "smoothed", "standard"),
        ("Figure_S17", "alt", "fixations", "smoothed", "standard"),
        ("Figure_S18", "altsdist", "fixations", "smoothed", "smoothed_raw"),
    ]:
        argv_target(name, "Figure_6_S13_S16_S17_S18.py",
                    [phase_space.format(paramset=paramset), out(name), mode, smoothed, layout])

    # -- supplementary figures with a __main__ -------------------------------------------
    argv_target("Figure_S3", "Figure_S3.py", [out("Figure_S3")])
    argv_target("Figure_S5", "Figure_S5.py", [out("Figure_S5")])
    argv_target("Figure_S14", "Figure_S14.py",
                [phase_space.format(paramset="default"), phase_space.format(paramset="alt"),
                 out("Figure_S14")])
    # The GWAS trait-variance figure (the former S15, from Table_S4.csv) was dropped from
    # the manuscript; its script is kept unbuilt as retired_Figure_S15_gwas_variance.py.
    argv_target("Figure_S15", "Figure_S15.py",
                [out("Figure_S15"), os.path.join(BASE, "Simons_2022_SSD_dfe.mat")])

    # -- Figure S4: S / sigma2 come from the rule's wildcards, not the script's defaults --
    S_S4, SIGMA2_S4 = "200", "40"
    name_s4 = f"Figure_S4__S{S_S4}_sigma2{SIGMA2_S4}"
    snake_target(
        name_s4, "Figure_S4.py",
        (lambda: SnakemakeShim(
            input=IOList(fixation_probability=expand(fixprob, S=S_S4, shift=SHIFT_101, sigma2=SIGMA2_S4)),
            output=IOList(figure_s4=out(name_s4)),
            params=Params(N=ns["N"], S=float(S_S4), sigma2=float(SIGMA2_S4)),
        )),
        [out(name_s4)])

    # -- Figure S6 ------------------------------------------------------------------------
    sdist = "expon"
    snake_target(
        f"Figure_S6__{sdist}", "Figure_S6.py",
        (lambda sdist=sdist: SnakemakeShim(
            input=IOList(
                simulations_A=[f for shift, sigma2 in COMBOS
                               for f in expand(sim, sdist=sdist, N2U=N2U_SWEEP,
                                               shift=shift, sigma2=sigma2)],
                analytic=[f for shift, sigma2 in COMBOS
                          for f in expand(analytic_fixed, sdist=sdist, shift=shift, sigma2=sigma2)],
            ),
            output=IOList(figure_s6=out(f"Figure_S6__{sdist}")),
            params=Params(sdist=SDISTS[sdist]),
        )),
        [out(f"Figure_S6__{sdist}")])

    # -- Figure S7: two outputs (fixations + contribution) per mutational input -----------
    for n2u in [ns["FIGURE_MAIN_N2U"], ns["FIGURE_ALT_N2U"]]:
        sdist, sigma2 = "expon", "40"
        fix_name = f"Figure_S7__{sdist}_{n2u}_{sigma2}"
        con_name = f"Figure_S7_contribution__{sdist}_{n2u}_{sigma2}"
        snake_target(
            fix_name, "Figure_S7.py",
            (lambda sdist=sdist, sigma2=sigma2, n2u=n2u, fix_name=fix_name, con_name=con_name:
             SnakemakeShim(
                 input=IOList(
                     simulations=expand(sim_s7, sdist=sdist, N2U=n2u, shift=SHIFT_42, sigma2=sigma2),
                     analytic=expand(analytic, sdist=sdist, shift=SHIFT_101, sigma2=sigma2),
                 ),
                 output=IOList(fixations=out(fix_name), contribution=out(con_name)),
                 params=Params(sdist=SDISTS[sdist], N2U=float(n2u), sigma2=float(sigma2)),
             )),
            [out(fix_name), out(con_name)])

    # -- Figures S8 / S9 / S10: the three-panel (shift, sigma2) supplements ---------------
    # S8 is drawn for both effect-size distributions (both exist in results/plots).
    for sdist in ["expon", "mixexpunif"]:
        name = f"Figure_S8__{sdist}"
        snake_target(
            name, "Figure_S8.py",
            (lambda sdist=sdist, name=name: SnakemakeShim(
                input=IOList(simulations=[f for shift, sigma2 in COMBOS
                                          for f in expand(sim, sdist=sdist, N2U=N2U_SWEEP,
                                                          shift=shift, sigma2=sigma2)]),
                output=IOList(figure_s8=out(name)),
                params=Params(sdist=SDISTS[sdist],
                              min_a=float(np.sqrt(SDISTS[sdist].ppf(0.0001)))),
            )),
            [out(name)])

    sdist = "expon"
    name = f"Figure_S9__{sdist}"
    snake_target(
        name, "Figure_S9.py",
        (lambda sdist=sdist, name=name: SnakemakeShim(
            input=IOList(
                simulations=[f for shift, sigma2 in COMBOS
                             for f in expand(sim, sdist=sdist, N2U=fnum(0.01),
                                             shift=shift, sigma2=sigma2)],
                analytic=[f for shift, sigma2 in COMBOS
                          for f in expand(analytic_fixed, sdist=sdist, shift=shift, sigma2=sigma2)],
            ),
            output=IOList(figure_s9=out(name)),
            params=Params(sdist=SDISTS[sdist]),
        )),
        [out(name)])

    name = f"Figure_S10__{sdist}"
    snake_target(
        name, "Figure_S10.py",
        (lambda sdist=sdist, name=name: SnakemakeShim(
            input=IOList(
                simulations=[f for shift, sigma2 in COMBOS
                             for f in expand(sim, sdist=sdist, N2U=N2U_SWEEP,
                                             shift=shift, sigma2=sigma2)],
                analytic=[f for shift, sigma2 in COMBOS
                          for f in expand(analytic_fixed, sdist=sdist, shift=shift, sigma2=sigma2)],
            ),
            output=IOList(figure_s10=out(name)),
            params=Params(sdist=SDISTS[sdist]),
        )),
        [out(name)])

    # -- Figure S11: the mixture distribution, per-effect-size-window panels --------------
    sdist = "mixexpunif"
    name = f"Figure_S11__{sdist}"
    snake_target(
        name, "Figure_S11.py",
        (lambda sdist=sdist, name=name: SnakemakeShim(
            input=IOList(
                simulations=[f for shift, sigma2 in COMBOS
                             for f in expand(sim, sdist=sdist, N2U=N2U_SWEEP,
                                             shift=shift, sigma2=sigma2)],
                analytic=[f for shift, sigma2 in COMBOS
                          for f in expand(analytic_window, sdist=sdist, shift=shift, sigma2=sigma2)],
            ),
            output=IOList(figure_s11=out(name)),
            params=Params(sdist=SDISTS[sdist]),
        )),
        [out(name)])

    # -- Figure S12 ------------------------------------------------------------------------
    snake_target(
        "Figure_S12", "Figure_S12.py",
        (lambda: SnakemakeShim(
            input=IOList(
                simulation_pickles=expand(
                    os.path.join(RESULTS, "single_site",
                                 "mode{mode}_S{S}_sign{sign}_shift{shift}_sigma2{sigma2}_Ssweep.pkl"),
                    mode="segregating", S=list(np.logspace(-3, 3, 31)), sign="pos",
                    shift=[80, 50], sigma2=[40, 80]),
                hayward_pickles=expand(
                    os.path.join(RESULTS, "hayward_fixation_probability",
                                 "S{S}_sign{sign}_shift{shift}_sigma2{sigma2}_Hayward.pkl"),
                    S=list(np.logspace(-3, 3, 200)), sign="pos", shift=[80, 50], sigma2=[40, 80]),
                double_recursion_pickles=expand(
                    fixprob, S=list(np.logspace(-3, 3, 200)), shift=[80, 50], sigma2=[40, 80]),
            ),
            output=IOList(figure_s12=out("Figure_S12")),
        )),
        [out("Figure_S12")])

    return targets


# --------------------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------------------

def check_inputs(shim):
    """Missing input files, as (attribute, path) pairs. Empty when the rule is satisfiable."""
    missing = []
    for key, value in vars(shim.input).items():
        for path in (value if isinstance(value, (list, tuple)) else [value]):
            if isinstance(path, str) and not os.path.exists(path):
                missing.append((key, path))
    return missing


def run_target(target, drop_missing):
    script = os.path.join(BASE, target["script"])
    if target["kind"] == "argv":
        missing = [a for a in target["argv"]
                   if a.endswith(".pkl") or a.endswith(".csv") or a.endswith(".mat")]
        missing = [m for m in missing if not os.path.exists(m)]
        if missing:
            raise FileNotFoundError(f"missing input(s): {missing}")
        sys.argv = [script] + list(target["argv"])
        runpy.run_path(script, run_name="__main__")
        return []

    shim = target["make"]()
    missing = check_inputs(shim)
    if missing and drop_missing:
        # Several sweeps were never simulated at the top of the 2NU range (see
        # results/plots/todo_claude.txt); the figures are built from whatever exists,
        # which is what the previous PNGs show.
        for key in {k for k, _ in missing}:
            value = getattr(shim.input, key)
            if isinstance(value, list):
                setattr(shim.input, key, [p for p in value if os.path.exists(p)])
    elif missing:
        raise FileNotFoundError(f"{len(missing)} missing input(s), first: {missing[0][1]}")

    # Run the script exactly the way Snakemake does: with `snakemake` already in its
    # globals, so the script's own `if "snakemake" in globals(): main(snakemake)` fires.
    sys.argv = [script]
    runpy.run_path(script, init_globals={"snakemake": shim}, run_name="__main__")
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ext", default=".tif", help="output extension (default: .tif)")
    parser.add_argument("--dpi", type=int, default=600, help="output resolution (default: 600)")
    parser.add_argument("--outdir", default=os.path.join(RESULTS, "plots_submission"))
    parser.add_argument("--only", default=None,
                        help="comma-separated figure names (or prefixes) to build")
    parser.add_argument("--list", action="store_true", help="list targets and exit")
    parser.add_argument("--strict", action="store_true",
                        help="fail instead of dropping input files that do not exist")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.chdir(BASE)
    sys.path.insert(0, BASE)

    ns = load_snakefile_header()
    targets = build_targets(ns, args.outdir, args.ext)

    if args.only:
        wanted = [w.strip() for w in args.only.split(",") if w.strip()]
        targets = [t for t in targets if any(t["name"].startswith(w) for w in wanted)]

    if args.list:
        for t in targets:
            print(f"{t['name']:52s} {t['script']}")
        return 0

    patch_savefig(args.dpi, args.ext)

    failures, notes = [], []
    for t in targets:
        print(f"[{t['name']}] building via {t['script']} ...", flush=True)
        try:
            missing = run_target(t, drop_missing=not args.strict)
            if missing:
                notes.append((t["name"], len(missing)))
                print(f"[{t['name']}] NOTE: {len(missing)} declared input file(s) absent, "
                      f"built from the rest (e.g. {os.path.basename(missing[0][1])})", flush=True)
            for path in t["outputs"]:
                size = os.path.getsize(path) / 1e6 if os.path.exists(path) else 0.0
                print(f"[{t['name']}] wrote {os.path.basename(path)} ({size:.1f} MB)", flush=True)
        except Exception as exc:
            failures.append((t["name"], f"{type(exc).__name__}: {exc}"))
            print(f"[{t['name']}] FAILED: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
        finally:
            import matplotlib.pyplot as plt
            plt.close("all")

    print("\n==== summary ====")
    print(f"targets: {len(targets)}  failed: {len(failures)}  "
          f"built with missing inputs: {len(notes)}")
    for name, count in notes:
        print(f"  partial: {name} ({count} inputs absent)")
    for name, err in failures:
        print(f"  FAILED : {name}  {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
