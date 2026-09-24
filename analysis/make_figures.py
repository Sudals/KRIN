"""Paper figures for the four-dataset, geo-graph, normal-vs-shock study.

Supersedes ``make_figures.py``, which predates every current decision (it is
airport-only, still plots the retired recovery regime, reads the mixed-graph
``main_results.csv``, and labels STGNN-R as "ours").

Terminology fixed here, and it should match the manuscript:
  * the model is **KRIN**; STGNN-R is a separate, earlier model of ours
  * the two conditions are **Normal** (``inregime``) and **Shock**
    (``cross_pre2shock``); the recovery regime is retired
  * every number is on the **geographic graph** ``A_geo``. the only graph built
    identically for all four datasets, hence the only fair cross-dataset basis
  * **MAE** is volume-weighted; **NMAE (macro)** normalises each node by its own
    level and then averages nodes equally, so it is the scale-controlled view.
    ``MAE_macro`` is deliberately not shown: with whole-node masking on a
    balanced panel it is arithmetically equal to MAE
  * the diagnostic is **|shift|**, the level shift in units of each node's
    training-time standard deviation. Dividing it by the in-train amplitude was
    tested in a pre-registered controlled sweep and did not improve alignment, so
    there is no denominator. The claim it supports is *whether* an out-of-support
    shift occurs, not the size of any margin

Every figure writes the numbers it plotted, standard deviations included, to a
sibling CSV, so each value in the paper can be traced back to a run file.

Output names carry no figure numbers on purpose: the draft was re-ordered once
already, and numbering lives in the manuscript, not in the filenames.

  fig_headline     accuracy per model, Normal vs Shock, with the HA and
                   SNaive-7 reference lines
  fig_support      system level against the band of levels seen in training
  fig_networks     the geographic graph per dataset, plus its statistics
  fig_degradation  Normal -> Shock error change per model
  fig_scale        KRIN's margin under MAE and under NMAE (macro)
  fig_transfer     every backbone before/after KRIN, with rank annotations
  fig_protob       the same, with 25% of the history removed, plus whether the
                   backbone ordering is reproducible across missing structures

Usage:  python3 analysis/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "legend.fontsize": 8.5, "figure.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.28, "axes.axisbelow": True,
    "savefig.bbox": "tight",
})

OUT = Path("results/figures")

# dataset order = increasing |shift| (in-support -> out-of-support)
#
# The 5th field is which SHOCK regime slice to read. Airports is the only dataset
# whose shock window spans two calendar years (2020-21) while the other three end
# in 2020, and its 2021 half is largely back to normal levels -- so reading its
# 24-month "all" row against the others' 12 months compares different things.
# ``shock2020`` is the length-matched slice, which is the whole point of having
# partitioned that window.
# Ordered by the diagnostic measured on the SAME window the errors come from.
# Airports is evaluated on shock2020, where |shift| is 3.845 rather than the
# 3.102 of the full 2020-21 span, which puts it after Chicago rather than before.
DATASETS = [
    ("Bikeshare DC", "configs/bike.yaml", "results_bike", "trips/day",   "all"),
    ("NYC Subway",   "configs/mta.yaml",  "results_mta",  "riders/day",  "all"),
    ("Chicago 'L'",  "configs/cta.yaml",  "results_cta",  "entries/day", "all"),
    ("Airports",     "configs/base.yaml", "results",      "flights/day", "shock2020"),
]
# The five published methods, plus the same SPIN backbone carrying KRIN. The
# graph network used as an internal control is excluded here so that counts
# quoted as "n of the published methods" have the right denominator.
# The suffix marks these as the simplified variants trained under the common
# protocol, not the authors' released implementations; the manuscript uses the
# same convention so a reader never has to work out which arm a bar came from.
MODELS = {
    "spin_c":  ("SPIN-s + KRIN", "#d62728", True),
    "ignnk":  ("IGNNK-s",       "#1f77b4", False),
    "satcn":  ("SATCN-s",       "#2ca02c", False),
    "spin":   ("SPIN-s",        "#bcbd22", False),
    "kits":   ("KITS-s",        "#17becf", False),
    "grin":   ("GRIN-s",        "#8c6d31", False),
}
# KRIN is the same wrapper applied uniformly to every backbone; ``spin_c`` is that
# wrapper on SPIN, plotted here for reference rather than as a proposed system. (``krinat_noa`` is an equivalent
# re-implementation and lands within training noise of it -- using the uniform
# wrapper keeps the transfer table internally consistent.)
OURS = "spin_c"
SCEN = {"inregime": "Normal", "cross_pre2shock": "Shock"}
RATIO = 0.5          # headline mask ratio


# A second copy is written flat, so a document can reference each figure by bare
# filename without the two ever drifting apart.
SUBMIT = Path("figures")


def _save(fig, name, table=None):
    OUT.mkdir(parents=True, exist_ok=True)
    SUBMIT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(SUBMIT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)
    if table is not None:
        table.to_csv(OUT / f"{name}.csv", index=False)
    print(f"  wrote {name}.pdf/.png" + ("/.csv" if table is not None else ""))


def load(res, shock_regime="all"):
    # geo3m10 = same protocol with 10 mask seeds (supersedes geo3). Fall back so
    # the script still runs on a dataset whose m10 pass has not finished.
    f = Path(res) / "main_results_geo3m10.csv"
    if not f.exists():
        f = Path(res) / "main_results_geo3.csv"
    frames = [pd.read_csv(f)]
    # the proposed model and the centring transfer live in their own run files
    for extra in ("main_results_prop.csv", "main_results_abl3.csv",
                  "main_results_ablate.csv", "main_results_ablate2.csv"):
        g = Path(res) / extra
        if g.exists():
            frames.append(pd.read_csv(g))
    a = pd.concat(frames, ignore_index=True)
    a = a[a.strategy == "random"]
    # Normal always reads "all"; Shock reads the dataset's length-matched slice.
    keep = ((a.scenario == "inregime") & (a.regime == "all")) | \
           ((a.scenario == "cross_pre2shock") & (a.regime == shock_regime))
    return a[keep]


def baseline(res, model, scen, ratio, col="MAE", shock_regime="all"):
    b = pd.read_csv(Path(res) / "baselines.csv")
    reg = shock_regime if scen == "cross_pre2shock" else "all"
    b = b[(b.model == model) & (b.strategy == "random") & (b.regime == reg)
          & (b.scenario == scen) & (np.isclose(b.ratio, ratio))]
    return b[col].mean()


def ha(res, scen, ratio, col="MAE", shock_regime="all"):
    return baseline(res, "HA", scen, ratio, col, shock_regime)


# --------------------------------------------------------------------------- #
# The held-out air-quality network is shown only here. It has no runs for the
# transfer, protocol-B or sweep figures, and it is the panel that carries the
# diagnostic's negative prediction, so this is where it belongs.
SUPPORT = ([("EPA NO2", "configs/aqs.yaml", "results_aqs", "ppb", "shock")]
           + list(DATASETS))


def paired_gap(res, regime, a, b, metric="MAE"):
    """Relative gap of ``a`` against ``b``, averaged over matched runs.

    Pairing on (train seed, mask seed) removes the variation both models share,
    which is why the appendix reports its intervals this way. Falls back to the
    ratio of means if the per-run files are not present.
    """
    frames = [pd.read_csv(Path(res) / f) for f in
              ("runs_geo3m10.csv", "runs_ablate.csv", "runs_ablate2.csv")
              if (Path(res) / f).exists()]
    if not frames:
        return np.nan
    d = pd.concat(frames, ignore_index=True)
    d = d[(d.strategy == "random") & (np.isclose(d.ratio, RATIO))
          & (d.scenario == "cross_pre2shock") & (d.regime == regime)]
    d = d.drop_duplicates(["model", "train_seed", "mask_seed"])
    ka = d[d.model == a].set_index(["train_seed", "mask_seed"])[metric]
    kb = d[d.model == b].set_index(["train_seed", "mask_seed"])[metric]
    idx = ka.index.intersection(kb.index)
    return float(((ka[idx] - kb[idx]) / kb[idx] * 100).mean())


def fig1_support(rows):
    """The mechanism: does the shock leave the span of levels seen in training?"""
    # Printed at \linewidth, so a canvas laid out in one row of five is shrunk to
    # about a third and every label with it. Two rows of three keep the scale
    # factor near a half, which is what makes the annotations readable.
    fig, grid = plt.subplots(2, 3, figsize=(9.6, 5.8), sharey=True)
    axes = list(grid.ravel())
    axes[-1].axis("off")                       # five panels in a 2x3 grid
    axes = axes[:len(SUPPORT)]
    tab = []
    for ax, (name, cfg_p, res, _unit, sreg) in zip(axes, SUPPORT):
        cfg = utils.load_config(cfg_p)
        proc = Path(cfg["paths"]["processed"])
        m = pd.read_parquet(proc / "traffic_matrix.parquet")
        dj = utils.load_json(proc / "dates.json")
        dates = pd.to_datetime(dj["dates"])
        sp = utils.load_json(proc / "splits.json")["scenarios"]["cross_pre2shock"]
        sc = utils.TrafficScaler.from_dict(sp["scaler"])
        S = sc.transform(m.values)
        raw_level = np.nanmean(S, axis=1)                 # statistic uses this
        level = pd.Series(raw_level, index=dates).rolling(7, min_periods=1).mean()
        tr = np.asarray(sp["train"] + sp["val"]); te = np.asarray(sp["test"])
        # Smoothing is for legibility only. The band and the below-band count are
        # computed on the daily series, since smoothing clips the training minimum
        # and would not match the diagnostic.
        lo, hi = np.nanmin(raw_level[tr]), np.nanmax(raw_level[tr])

        ax.axhspan(lo, hi, color="#4c72b0", alpha=0.15, lw=0)
        ax.axhline(lo, color="#4c72b0", lw=0.9, ls="--")
        # The band is [min, max] of the DAILY series, but a 7-day smooth pulls the
        # extremes inwards by up to 2.9 units, so a smoothed-only panel shows a band
        # whose edges no visible curve ever reaches. Draw the daily series faintly
        # underneath so the band is visibly supported by the data that defines it , 
        # and so the reader can see that the amplitude denominator is set by a few
        # single-day extremes.
        ax.plot(dates[tr], raw_level[tr], color="#4c72b0", lw=0.4, alpha=0.3)
        ax.plot(dates[te], raw_level[te], color="#c44e52", lw=0.4, alpha=0.3)
        ax.plot(dates[tr], level.values[tr], color="#4c72b0", lw=1.1, label="train (normal)")
        ax.plot(dates[te], level.values[te], color="#c44e52", lw=1.1, label="test (shock)")
        if name not in rows:
            import shift_amp
            dd = shift_amp.diagnose(cfg_p, sreg)
            rows[name] = {"shift": dd["shift"], "amp": dd["amp"],
                          "ratio": abs(dd["shift"]), "window": sreg,
                          "below": dd["test_days_below_train_min_%"]}
        d = rows[name]
        # The panel draws the whole evaluation span so the 2021 recovery stays
        # visible, but the annotation is counted on the window the errors use,
        # which for airports is 2020 alone.
        te_stat = te
        if sreg not in (None, "all"):
            regs = np.array(dj["regime"])
            yr = sreg[-4:]
            base = sreg[:-4] if yr.isdigit() else sreg
            sel = [t for t in te if regs[t] == base
                   and (not yr.isdigit() or dates[t].year == int(yr))]
            if sel:
                te_stat = np.asarray(sel, dtype=int)
        below = float(np.mean(raw_level[te_stat] < lo) * 100)
        # Only a year-restricted window needs saying; "shock" already is the
        # whole evaluation span for the panels that use it.
        span = f" ({sreg[-4:]})" if sreg not in (None, "all") \
            and sreg[-4:].isdigit() else ""
        # Stats go inside the axes: as a two-line title they ran into the
        # neighbouring panel at this figure width.
        ax.set_title(name, fontsize=13)
        ax.text(0.03, 0.06,
                f"|shift| {abs(d['shift']):.2f}{span}"
                f"\n{below:.0f}% below band",
                transform=ax.transAxes, fontsize=11.5, va="bottom",
                bbox=dict(fc="white", ec="0.7", alpha=0.92, pad=3.5))
        # Airports is the only dataset whose shock window runs two calendar years;
        # its 2021 half is largely back to normal, so mark the boundary rather than
        # let the panel read as one homogeneous condition.
        if name == "Airports":
            ax.axvline(pd.Timestamp("2021-01-01"), color="0.35", lw=0.8, ls=":")
            ax.text(pd.Timestamp("2021-02-01"), ax.get_ylim()[0] * 0.62, "2021",
                    fontsize=7.5, color="0.35")
        import matplotlib.dates as _md
        ax.xaxis.set_major_locator(_md.YearLocator(2 if name == "Airports" else 1))
        ax.xaxis.set_major_formatter(_md.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=0, labelsize=11)
        ax.tick_params(axis="y", labelsize=11)
        tab.append({"dataset": name, "shift": d["shift"], "amplitude": d["amp"],
                    "abs_shift": abs(d["shift"]),
                    "shock_days_below_train_min_pct": below})
    for a in (axes[0], axes[3]):
        a.set_ylabel("system level\n(train-standardised)", fontsize=11)
    # The sixth cell of the grid is empty, so the legend lives there rather than
    # floating over the Airports panel.
    spare = grid.ravel()[-1]
    h, l = axes[0].get_legend_handles_labels()
    spare.legend(h, l, loc="center", ncol=1, framealpha=0.9, fontsize=11,
                 title="pale = daily, bold = 7-day mean;\nthe band and every statistic\n"
                       "use the daily series",
                 title_fontsize=11)
    fig.suptitle("Shaded band = range of system levels seen during training; |shift| is the level\n"
                 "shift in units of each node's training standard deviation. The first two panels\n"
                 "stay inside the band and nothing collapses there; the last three leave it below\n"
                 "and 5 or 6 of 6 backbones collapse. Panels run in order of the diagnostic.",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    _save(fig, "fig_support", pd.DataFrame(tab))


def fig2_main(order):
    """Headline accuracy, Normal vs Shock, one panel per dataset."""
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.5))
    tab = []
    for ax, (name, _c, res, unit, sreg) in zip(axes, DATASETS):
        a = load(res, sreg)
        names, norm, shock = [], [], []
        for mk, (lbl, col, ours) in MODELS.items():
            n_ = a[(a.model == mk) & (a.scenario == "inregime") & np.isclose(a.ratio, RATIO)]
            s_ = a[(a.model == mk) & (a.scenario == "cross_pre2shock") & np.isclose(a.ratio, RATIO)]
            names.append(lbl); norm.append(n_.MAE_mean.iloc[0]); shock.append(s_.MAE_mean.iloc[0])
            tab.append({"dataset": name, "model": lbl,
                        "normal_MAE": n_.MAE_mean.iloc[0], "normal_std": n_.MAE_std.iloc[0],
                        "shock_MAE": s_.MAE_mean.iloc[0], "shock_std": s_.MAE_std.iloc[0]})
        y = np.arange(len(names))
        ax.barh(y - 0.2, norm, 0.38, color="#93b3d6", label="Normal")
        ax.barh(y + 0.2, shock, 0.38, color="#c44e52", label="Shock")
        for i, mk in enumerate(MODELS):
            if MODELS[mk][2]:
                ax.barh(y[i] - 0.2, norm[i], 0.38, color="#93b3d6", edgecolor="k", lw=1.1)
                ax.barh(y[i] + 0.2, shock[i], 0.38, color="#c44e52", edgecolor="k", lw=1.1)
        ax.axvline(ha(res, "cross_pre2shock", RATIO, shock_regime=sreg), color="k", ls=":", lw=1.2,
                   label="HA (shock)")
        p7 = baseline(res, "SNaive-7", "cross_pre2shock", RATIO, shock_regime=sreg)
        ax.axvline(p7, color="#7f7f7f", ls="--", lw=1.2, label="SNaive-7 (shock)")
        ax.set_yticks(y)
        ax.set_yticklabels(names if ax is axes[0] else [], fontsize=8.5)
        ax.invert_yaxis(); ax.set_xlabel(f"MAE ({unit})")
        ax.set_title(name, fontsize=10.5)
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="upper center", bbox_to_anchor=(2.25, -0.22), ncol=4,
                   framealpha=0.95)
    fig.suptitle(f"Reconstruction error at r = {RATIO} hidden nodes, geographic graph. "
                 "Reference lines: historical average (dotted) and SNaive-7,\n"
                 "i.e. copying the hidden node's own value from a week earlier (dashed). "
                 "a zero-parameter competitor most models fail to beat under shock.",
                 y=1.10, fontsize=10)
    _save(fig, "fig_headline", pd.DataFrame(tab))


def fig3_degradation(order):
    """What the shift does to each model, relative to its own normal-condition error."""
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    tab = []
    width = 0.2
    x = np.arange(len(MODELS))
    for k, (name, _c, res, _u, sreg) in enumerate(DATASETS):
        a = load(res, sreg); vals = []
        for mk in MODELS:
            n_ = a[(a.model == mk) & (a.scenario == "inregime") & np.isclose(a.ratio, RATIO)].MAE_mean.iloc[0]
            s_ = a[(a.model == mk) & (a.scenario == "cross_pre2shock") & np.isclose(a.ratio, RATIO)].MAE_mean.iloc[0]
            vals.append((s_ / n_ - 1) * 100)
            tab.append({"dataset": name, "model": MODELS[mk][0], "degradation_pct": vals[-1]})
        ax.bar(x + (k - 1.5) * width, vals, width, label=name)
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels([MODELS[m][0] for m in MODELS], rotation=25, ha="right")
    ax.set_ylabel("error change, Normal → Shock (%)")
    ax.set_title("Degradation tracks whether the shift leaves training support: "
                 "flat on bikeshare, large on the other three")
    ax.legend(ncol=4, framealpha=0.95)
    _save(fig, "fig_degradation", pd.DataFrame(tab))


def fig4_scale_control(order):
    """The metric axis: volume-weighted MAE vs the scale-controlled NMAE (macro).

    Datasets sit on a categorical axis ordered by the diagnostic, with no line
    joining them. A connecting line would assert a monotone relationship, and the
    lack of one is what these numbers show.
    """
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    tab, xs = [], np.arange(len(DATASETS))
    bars = {}
    for off, (met, col, lbl) in enumerate(
            (("MAE", "#4c72b0", "MAE (volume-weighted)"),
             ("NMAE_macro", "#dd8452", "NMAE, macro (scale-controlled)"))):
        ys = []
        for name, cfg_p, res, _u, sreg in DATASETS:
            a = load(res, sreg)
            c = a[(a.scenario == "cross_pre2shock") & np.isclose(a.ratio, RATIO)
                  & a.model.isin(MODELS)]
            k = c[c.model == OURS].iloc[0]
            o = c[c.model != OURS]
            nb = o.loc[o[f"{met}_mean"].idxmin()]
            # Paired over (train seed, mask seed), the same estimator appendix
            # A.2.2 attaches its intervals to. A ratio of the two means instead
            # would disagree with the text by a point or two on the two deep
            # collapses, for no gain.
            marg = paired_gap(res, sreg, OURS, nb.model, met)
            ys.append(marg)
            tab.append({"dataset": name, "metric": met, "krin": k[f"{met}_mean"],
                        "next_best_model": nb.model, "next_best": nb[f"{met}_mean"],
                        "margin_pct": marg, "shift_over_amp": order[name]["ratio"]})
        bars[met] = ys
        ax.bar(xs + (off - 0.5) * 0.36, ys, 0.36, color=col, label=lbl)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:+.0f}%", (x + (off - 0.5) * 0.36, y),
                        textcoords="offset points", xytext=(0, -13 if y < 0 else 5),
                        ha="center", fontsize=9)
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(xs)
    # The diagnostic printed under each name is the one measured on that
    # dataset's own evaluation window, so the axis ordering and the errors above
    # it refer to the same days.
    ax.set_xticklabels([f"{n}\n(|shift| {order[n]['ratio']:.2f})"
                        for n, *_r in DATASETS], fontsize=9.5)
    ax.set_ylabel("SPIN-s + KRIN vs best uncentred backbone (%)\n"
                  "negative favours KRIN", fontsize=9.5)
    # No in-figure title: the caption carries the reading, and the bars hang from
    # y=0 at the top so any text inside the axes lands on Subway's value labels.
    # The legend goes just outside, above the frame, where nothing collides.
    ax.legend(framealpha=0.95, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=2, fontsize=9, frameon=False, borderaxespad=0.0)
    ax.margins(y=0.14)
    _save(fig, "fig_scale", pd.DataFrame(tab))


def _log_ticks(ax, lo_vals, hi_vals):
    """A log axis whose data spans less than one decade gets no major tick inside
    the view, so the panel ends up with no numbers at all (bikeshare, range ~6 to
    10). Labelling the minor ticks instead crowds the wider panels (subway, where
    600/700/800/900/1000 run together). Pick four round values across the actual
    data range whenever the default majors are too few."""
    lo, hi = np.nanmin(lo_vals), np.nanmax(hi_vals)
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0 or hi <= lo:
        return
    decades = [10.0 ** d for d in
               range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)]
    if sum(lo <= d <= hi for d in decades) >= 3:
        return                                  # matplotlib's own ticks are fine
    t = np.unique([float(f"{v:.2g}") for v in np.geomspace(lo, hi, 4)])
    ax.set_xticks(t)
    ax.set_xticklabels([f"{v:g}" for v in t], fontsize=7.5)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def fig_transfer():
    """The centring applied to all six reimplemented backbones, and what it does to
    their ORDER. Uncentred, the expressive aggregator (SPIN attention) ranks 3rd/3rd/2nd on
    the three out-of-support datasets; centred it ranks 1st on all three. Where there
    is no shift (bikeshare) the order is preserved. So the field's conclusions about
    which aggregator is better are confounded by a normalisation nobody applied."""
    # (uncentred name, centred name, label). STGNN is a plain spatio-temporal
    # graph network, included to show the operation is not specific to published
    # methods. Its centred counterpart is ``stgnn_c``: the same zero-parameter
    # wrapper every other row uses, so capacity stays exactly equal within a row.
    # The suffix marks a simplified variant of a published method; STGNN is this
    # paper's own control on the same protocol, so it carries no suffix.
    PUB = [("ignnk", "ignnk_c", "IGNNK-s"), ("satcn", "satcn_c", "SATCN-s"),
           ("spin", "spin_c", "SPIN-s"), ("kits", "kits_c", "KITS-s"),
           ("grin", "grin_c", "GRIN-s"), ("stgnn", "stgnn_c", "STGNN")]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.6))
    tab = []
    for ax, (name, _c, res, unit, sreg) in zip(axes, DATASETS):
        a = load(res, sreg)
        s = a[(a.scenario == "cross_pre2shock") & np.isclose(a.ratio, RATIO)]
        pre = [s[s.model == a].MAE_mean.iloc[0] if (s.model == a).any() else np.nan
               for a, _, _ in PUB]
        post = [s[s.model == c].MAE_mean.iloc[0] if (s.model == c).any() else np.nan
                for _, c, _ in PUB]
        y = np.arange(len(PUB))
        ax.barh(y - 0.2, pre, 0.38, color="#c44e52", label="without KRIN")
        ax.barh(y + 0.2, post, 0.38, color="#4c72b0", label="+ KRIN")
        p7 = baseline(res, "SNaive-7", "cross_pre2shock", RATIO, shock_regime=sreg)
        ax.axvline(p7, color="#7f7f7f", ls="--", lw=1.2, label="SNaive-7")
        # mark the rank each method takes before and after
        rk_pre = {i: r + 1 for r, i in enumerate(np.argsort(np.nan_to_num(pre, nan=9e9)))}
        rk_post = {i: r + 1 for r, i in enumerate(np.argsort(np.nan_to_num(post, nan=9e9)))}
        for i, (_a, _c, lbl) in enumerate(PUB):
            ax.text(0.985, (len(PUB) - 1 - i + 0.5) / len(PUB),
                    f"{rk_pre[i]}\u2192{rk_post[i]}", transform=ax.transAxes,
                    ha="right", va="center", fontsize=7.5,
                    color="#d62728" if rk_pre[i] != rk_post[i] else "0.55",
                    # the SNaive-7 rule can pass straight through these labels on
                    # a narrow panel (bikeshare), so give them an opaque backing
                    bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))
            tab.append({"dataset": name, "model": lbl, "without_krin": pre[i],
                        "centred": post[i], "rank_before": rk_pre[i],
                        "rank_after": rk_post[i]})
        ax.set_yticks(y); ax.set_yticklabels([l for *_, l in PUB] if ax is axes[0] else [],
                                             fontsize=9)
        ax.invert_yaxis(); ax.set_xscale("log"); ax.set_xlabel(f"MAE ({unit}), log")
        # default log tick labels collide on the narrow-range panels
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(axis="x", labelsize=8)
        _log_ticks(ax, post, pre)
        ax.set_title(name, fontsize=10.5); ax.grid(axis="y", visible=False)
    axes[0].legend(loc="upper center", bbox_to_anchor=(2.25, -0.24), ncol=3,
                   framealpha=0.95)
    fig.suptitle("KRIN repairs every backbone, and reorders them. Numbers at the right "
                 "are each backbone's rank before\u2192after; red = the order changed.\n"
                 "SPIN-s, the most expressive aggregator, ranks 3rd, 4th and 4th uncentred "
                 "on the three datasets whose level leaves training support, and 1st on all "
                 "three once centred. Where nothing collapses (bikeshare) it was already 1st "
                 "and half the ranks are unchanged.",
                 y=1.10, fontsize=10)
    _save(fig, "fig_transfer", pd.DataFrame(tab))


def fig_protob():
    """The same reordering under a DIFFERENT missing structure.

    The rank inversion reported in the main text was measured where the hidden node's own history is
    complete. Protocol A cannot host the question at all (with no values ever,
    the level mu is undefined), but protocol B can, because a node observed at
    some timesteps still has a computable level. Here 25% of the history entries
    are removed and the measurement is repeated. Holes are filled with the node's
    mean over the entries it kept in that window, which makes the filled window
    mean exactly equal the observed-only mean -- filling with 0 would drag the
    level toward the node's global mean and silently disable the very operation
    under test. The same hole mask is used for every model."""
    PUB = [("ignnk", "IGNNK-s"), ("satcn", "SATCN-s"), ("spin", "SPIN-s"),
           ("kits", "KITS-s"), ("grin", "GRIN-s"), ("stgnn", "STGNN")]
    from scipy.stats import spearmanr
    fig = plt.figure(figsize=(19.5, 3.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.05], wspace=0.28)
    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
    tab, rho = [], []
    for ax, (name, _c, res, unit, sreg) in zip(axes[:4], DATASETS):
        h = pd.read_csv(Path(res) / "main_results_protoB.csv")
        h = h[(h.strategy == "random") & (h.regime == sreg)
              & np.isclose(h.ratio, RATIO)].set_index("model").MAE_mean
        pre = [h.get(m, np.nan) for m, _ in PUB]
        post = [h.get(m + "_c", np.nan) for m, _ in PUB]
        y = np.arange(len(PUB))
        ax.barh(y - 0.2, pre, 0.38, color="#c44e52", label="without KRIN")
        ax.barh(y + 0.2, post, 0.38, color="#4c72b0", label="+ KRIN")
        rk_pre = {i: r + 1 for r, i in enumerate(np.argsort(np.nan_to_num(pre, nan=9e9)))}
        rk_post = {i: r + 1 for r, i in enumerate(np.argsort(np.nan_to_num(post, nan=9e9)))}
        for i, (_m, lbl) in enumerate(PUB):
            ax.text(0.985, (len(PUB) - 1 - i + 0.5) / len(PUB),
                    f"{rk_pre[i]}\u2192{rk_post[i]}", transform=ax.transAxes,
                    ha="right", va="center", fontsize=7.5,
                    color="#d62728" if rk_pre[i] != rk_post[i] else "0.55",
                    # the SNaive-7 rule can pass straight through these labels on
                    # a narrow panel (bikeshare), so give them an opaque backing
                    bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))
            tab.append({"dataset": name, "model": lbl, "without_krin": pre[i],
                        "centred": post[i], "rank_before": rk_pre[i],
                        "rank_after": rk_post[i]})
        # rank agreement with the complete-history condition of fig_transfer
        c = load(res, sreg)
        c = c[(c.scenario == "cross_pre2shock") & np.isclose(c.ratio, RATIO)] \
            .set_index("model").MAE_mean
        rho.append((name,
                    spearmanr([c.get(m, np.nan) for m, _ in PUB], pre).statistic,
                    spearmanr([c.get(m + "_c", np.nan) for m, _ in PUB], post).statistic))
        ax.set_yticks(y)
        ax.set_yticklabels([l for _, l in PUB] if ax is axes[0] else [], fontsize=9)
        ax.invert_yaxis(); ax.set_xscale("log"); ax.set_xlabel(f"MAE ({unit}), log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(axis="x", labelsize=8)
        _log_ticks(ax, post, pre)
        ax.set_title(name, fontsize=10.5); ax.grid(axis="y", visible=False)
    # fifth panel: is the ORDER itself reproducible across missing structures?
    ax = axes[4]
    x = np.arange(len(rho))
    ax.bar(x - 0.2, [r[1] for r in rho], 0.38, color="#c44e52", label="without KRIN")
    ax.bar(x + 0.2, [r[2] for r in rho], 0.38, color="#4c72b0", label="+ KRIN")
    ax.set_xticks(x); ax.set_xticklabels([r[0].split()[0] for r in rho],
                                         fontsize=8, rotation=18)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Spearman rho", fontsize=9)
    ax.set_title("Is the ranking reproducible?\ncomplete vs holey history", fontsize=9.5)
    ax.grid(axis="x", visible=False)
    for xi, r in zip(x, rho):
        ax.text(xi - 0.30, r[1] + 0.02, f"{r[1]:.2f}", ha="center", fontsize=7)
        ax.text(xi + 0.30, r[2] + 0.02, f"{r[2]:.2f}", ha="center", fontsize=7)
    axes[0].legend(loc="upper center", bbox_to_anchor=(2.3, -0.24), ncol=2,
                   framealpha=0.95)
    fig.suptitle("The reordering is not an artefact of complete history. With 25% of the "
                 "history removed, +KRIN still wins every one of the 24 cells and SPIN-s "
                 "still takes 1st on all four datasets\n(5\u21921 Chicago, 4\u21921 airports, "
                 "3\u21921 subway). Right: without centring the backbone ORDER is not "
                 "reproducible across missing structures; with it, it is.",
                 y=1.13, fontsize=10)
    _save(fig, "fig_protob", pd.DataFrame(tab))


def fig5_geo_graphs():
    """The geographic graph actually used, drawn per dataset, with its statistics.

    A_geo is a Gaussian kernel on haversine distance, sparsified to the top-k
    neighbours and symmetrised. built identically for all four datasets, which
    is what makes them comparable. The panels show it is a genuine spatial
    neighbourhood graph in each case and not, say, a near-complete graph on the
    small networks.
    """
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 6.4),
                             gridspec_kw={"height_ratios": [2.6, 1]})
    tab = []
    for k, (name, cfg_p, _res, _u, _sr) in enumerate(DATASETS):
        cfg = utils.load_config(cfg_p)
        proc = Path(cfg["paths"]["processed"])
        ni = utils.load_json(proc / "node_index.json")
        A = np.load(proc / "A_geo.npy")
        lat = np.array([v["lat"] for v in ni.values()])
        lon = np.array([v["lon"] for v in ni.values()])
        vol = np.array([v["mean_traffic"] for v in ni.values()])

        R = 6371.0
        la, lo_ = np.radians(lat)[:, None], np.radians(lon)[:, None]
        h = (np.sin((la - la.T) / 2) ** 2
             + np.cos(la) * np.cos(la.T) * np.sin((lo_ - lo_.T) / 2) ** 2)
        D = 2 * R * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
        ii, jj = np.where(np.triu(A) > 0)
        edge_km = D[ii, jj]
        deg = (A > 0).sum(1)

        ax = axes[0, k]
        for i, j in zip(ii, jj):
            ax.plot([lon[i], lon[j]], [lat[i], lat[j]], color="#8c8c8c",
                    lw=0.3, alpha=0.55, zorder=1)
        ax.scatter(lon, lat, s=5 + 45 * vol / vol.max(), c="#c44e52",
                   edgecolor="none", alpha=0.85, zorder=2)
        ax.set_title(f"{name}   N={len(ni)}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_aspect("equal", adjustable="datalim")
        ax.text(0.02, 0.02,
                f"mean degree {deg.mean():.1f}\nsparsity {1 - (A > 0).mean():.3f}\n"
                f"median edge {np.median(edge_km):.1f} km",
                transform=ax.transAxes, fontsize=7.5, va="bottom",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))

        ax2 = axes[1, k]
        ax2.hist(edge_km, bins=30, color="#4c72b0", alpha=0.85)
        ax2.set_xlabel("edge length (km)", fontsize=8.5)
        ax2.tick_params(labelsize=7.5)
        if k == 0:
            ax2.set_ylabel("edges", fontsize=8.5)
        tab.append({"dataset": name, "N": len(ni), "mean_degree": deg.mean(),
                    "sparsity": 1 - (A > 0).mean(),
                    "median_edge_km": float(np.median(edge_km)),
                    "p95_edge_km": float(np.percentile(edge_km, 95)),
                    "max_edge_km": float(edge_km.max())})
    fig.suptitle("The geographic graph A_geo used for every result: Gaussian kernel on "
                 "haversine distance, top-k sparsified, symmetrised.\n"
                 "Marker area \u221d mean traffic. Built identically across datasets, "
                 "which is what makes them comparable.", y=1.02, fontsize=10)
    fig.tight_layout()
    _save(fig, "fig_networks", pd.DataFrame(tab))


def main():
    order = {}
    import shift_amp
    for name, cfg_p, _res, _u, sreg in DATASETS:
        # Diagnose the window the errors are measured on, not the whole span.
        d = shift_amp.diagnose(cfg_p, sreg)
        order[name] = {"shift": d["shift"], "amp": d["amp"],
                       "ratio": abs(d["shift"]), "window": sreg,
                       "below": d["test_days_below_train_min_%"]}
        print(f"  [diag] {name:<14} ({sreg}) |shift| = {abs(d['shift']):.3f}  "
              f"below = {d['test_days_below_train_min_%']:.1f}%")
    print(f"\n[figures] -> {OUT}")
    fig1_support(order)
    fig2_main(order)
    fig3_degradation(order)
    fig4_scale_control(order)
    fig_transfer()
    fig_protob()
    fig5_geo_graphs()
    print("[done]")


if __name__ == "__main__":
    main()
