"""Publication-style figures (PNG 300 dpi + vector PDF). Okabe-Ito colour-blind-safe palette."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROPOSED = "SoilCastNet"
OI = ["#0072B2", "#E69F00", "#009E73", "#56B4E9", "#CC79A7", "#999999", "#F0E442", "#8C564B"]
EXTRA = OI + ["#332288", "#882255", "#44AA99", "#117733"]   # 12 distinct hues for many baselines
INK, MUTED, GRID = "#1f2933", "#6b7280", "#e5e7eb"
VERM = "#D55E00"                         # proposed model highlight


def style():
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "font.family": "DejaVu Sans"})


def colors(names):
    base = [n for n in names if not n.startswith(PROPOSED)]
    m = {n: EXTRA[i % len(EXTRA)] for i, n in enumerate(base)}
    for n in names:
        if n.startswith(PROPOSED): m[n] = VERM
    return m


def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig: dataset overview
def fig_dataset(groups, outdir, n_show=3):
    names = sorted(groups)[:n_show]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [2.3, 1]})
    for i, g in enumerate(names):
        d = groups[g]["target"]; ax[0].plot(d.index, d.values, lw=0.7, color=OI[i], label=str(g)[:18])
    ax[0].set(title="Soil-moisture records of three sensors", ylabel="Volumetric water content"); ax[0].legend(ncol=3, loc="upper right", fontsize=7)
    allv = np.concatenate([g["target"].dropna().values for g in groups.values()])
    ax[1].hist(allv, bins=60, color=OI[0], alpha=0.85); ax[1].set(title="Distribution (all sensors)", xlabel="VWC", ylabel="Count")
    save(fig, outdir, "fig01_dataset_overview")


# ---------------------------------------------------------------- Fig: architecture
def fig_architecture(outdir):
    fig, ax = plt.subplots(figsize=(11, 3.6)); ax.axis("off"); ax.set_xlim(0, 11.1); ax.set_ylim(0, 3.6)
    def box(x, y, w, h, t, fc="#e8f1fa", ec=OI[0], fs=7.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=INK)
    def arr(x1, y1, x2, y2): ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, color=MUTED, lw=1))
    box(0.1, 1.4, 1.35, 0.9, "Input window\n(L steps x F feats:\nVWC, temp, ΔVWC,\ntime encodings)", "#f4f4f4", MUTED)
    for i, d in enumerate((1, 2, 4, 8)):
        box(1.95, 2.75 - i * 0.78, 1.5, 0.6, f"Causal conv, dilation {d}"); arr(1.45, 1.85, 1.95, 3.05 - i * 0.78)
        arr(3.45, 3.05 - i * 0.78, 3.9, 1.85)
    box(3.9, 1.4, 1.15, 0.9, "Concat +\nLinear +\nLayerNorm"); arr(5.05, 1.85, 5.35, 1.85)
    box(5.35, 1.4, 1.1, 0.9, "GRU +\ntemporal\nattention"); arr(6.45, 1.85, 6.8, 1.85)
    box(6.8, 1.4, 0.8, 0.9, "Latent z", "#fdf0d5", OI[1])
    heads = [("decay λ_h", 3.0), ("recharge r_h", 2.25), ("event gate g_h", 1.5), ("width lo/hi", 0.75)]
    for t, y in heads:
        box(7.95, y - 0.27, 1.35, 0.5, t, "#e5f5ee", OI[2]); arr(7.6, 1.85, 7.95, y)
        arr(9.3, y, 9.65, 1.85)
    box(9.65, 1.1, 1.3, 1.5, "Median =\na − λ(a−floor) + g·r\n\nq05/q95 =\nmedian ∓ width\n\n→ CQR calibration", "#fde6dc", VERM, 6.3)
    ax.text(5.5, 0.15, "Physics-guided loss: pinball + range penalty + 'no rise unless gate opens' + gate BCE (observed wetting events)", ha="center", fontsize=8, color=MUTED)
    ax.set_title("SoilCastNet architecture", loc="left")
    save(fig, outdir, "fig02_architecture")


# ---------------------------------------------------------------- Fig: training curves
def fig_training(histories, outdir):
    cm = colors(list(histories)); fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), sharey=False)
    for n, hs in histories.items():
        h = hs[0]; lw = 2.2 if n.startswith(PROPOSED) else 1.1
        ax[0].plot(h[:, 0], color=cm[n], lw=lw, label=n); ax[1].plot(h[:, 1], color=cm[n], lw=lw)
    ax[0].set(title="Training loss", xlabel="Epoch", ylabel="Pinball loss (z-units)"); ax[1].set(title="Validation loss", xlabel="Epoch")
    ax[0].legend(fontsize=6.5, ncol=2); save(fig, outdir, "fig03_training_curves")


# ---------------------------------------------------------------- Fig: model comparison
def fig_compare(summary, horizons, outdir, metric="MAE"):
    """summary: dict model -> {horizon_idx: (mean, std)} for metric."""
    names = list(summary); cm = colors(names)
    fig, ax = plt.subplots(1, len(horizons), figsize=(3.3 * len(horizons), 3.4), sharey=False)
    ax = np.atleast_1d(ax)
    for j, h in enumerate(horizons):
        order = sorted(names, key=lambda n: summary[n][j][0])
        m = [summary[n][j][0] for n in order]; s = [summary[n][j][1] for n in order]
        ax[j].barh(range(len(order)), m, xerr=s, color=[cm[n] for n in order], height=0.62, capsize=2, ecolor=MUTED)
        ax[j].set_yticks(range(len(order))); ax[j].set_yticklabels(order if j == 0 else [], fontsize=7.5); ax[j].invert_yaxis()
        ax[j].set(title=f"Horizon +{h}", xlabel=metric); ax[j].grid(axis="y", visible=False)
        for k, v in enumerate(m): ax[j].text(v, k, f" {v:.4f}", va="center", fontsize=6.5, color=INK)
    save(fig, outdir, f"fig04_model_comparison_{metric}")


def fig_metric_vs_horizon(table, horizons, outdir, metric="MAE"):
    names = list(table); cm = colors(names); fig, ax = plt.subplots(figsize=(5.2, 3.5))
    for n in names:
        m = [table[n][j][0] for j in range(len(horizons))]; s = [table[n][j][1] for j in range(len(horizons))]
        lw = 2.4 if n.startswith(PROPOSED) else 1.1
        ax.errorbar(horizons, m, yerr=s, color=cm[n], lw=lw, marker="o", ms=4, capsize=2, label=n)
    ax.set(xlabel="Forecast horizon (steps ahead)", ylabel=metric, title=f"{metric} vs. horizon"); ax.legend(fontsize=6.5, ncol=2)
    save(fig, outdir, f"fig05_{metric}_vs_horizon")


# ---------------------------------------------------------------- Fig: forecasts
def fig_forecast_ts(t_axis, obs, med, lo, hi, gate, event, outdir, h_label, n=500):
    n = min(n, len(obs)); sl = slice(0, n); x = np.arange(n)
    fig, ax = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.1]})
    ax[0].fill_between(x, lo[sl], hi[sl], color=VERM, alpha=0.2, lw=0, label="90% calibrated interval")
    ax[0].plot(x, obs[sl], color=INK, lw=1.2, label="Observed"); ax[0].plot(x, med[sl], color=VERM, lw=1.3, label="SoilCastNet median")
    ax[0].set(ylabel="Soil moisture (VWC)", title=f"Forecast at +{h_label} (consecutive test windows of one sensor)"); ax[0].legend(ncol=3, fontsize=7.5, loc="upper right")
    if gate is not None and not np.all(np.isnan(gate)):
        ax[1].plot(x, gate[sl], color=OI[2], lw=1, label="Gate probability g")
        ev = np.where(event[sl] > 0)[0]; ax[1].vlines(ev, 0, 1, color=OI[0], alpha=0.25, lw=0.8, label="Observed wetting event")
        ax[1].set(ylabel="Event gate", xlabel="Test window index", ylim=(0, 1.02)); ax[1].legend(fontsize=7, ncol=2, loc="upper right")
    save(fig, outdir, "fig06_forecast_with_interval")


def fig_scatter(obs, preds, outdir, h_label):
    names = list(preds); n = len(names); cols = min(4, n); rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(2.7 * cols, 2.7 * rows), squeeze=False)
    lo, hi = np.percentile(obs, [0.5, 99.5])
    for a, nm in zip(ax.ravel(), names):
        a.hexbin(obs, preds[nm], gridsize=45, extent=(lo, hi, lo, hi), bins="log", cmap="Blues", mincnt=1)
        a.plot([lo, hi], [lo, hi], color=VERM if nm.startswith(PROPOSED) else MUTED, lw=1)
        a.set_title(nm, fontsize=8); a.set_xlabel("Observed"); a.set_ylabel("Predicted"); a.grid(False)
    for a in ax.ravel()[n:]: a.axis("off")
    fig.suptitle(f"Predicted vs observed, horizon +{h_label}", y=1.0, fontsize=10); fig.tight_layout()
    save(fig, outdir, "fig07_scatter")


# ---------------------------------------------------------------- Fig: calibration
def fig_calibration(cal, outdir):
    """cal: dict model -> (raw PICP, conformal-calibrated PICP), mean over horizons, nominal 90 %."""
    names = list(cal); cm = colors(names); y = np.arange(len(names)); fig, ax = plt.subplots(figsize=(5.4, 0.36 * len(names) + 1.3))
    ax.barh(y - 0.19, [cal[n][0] for n in names], height=0.36, color=OI[5], label="Raw quantiles")
    ax.barh(y + 0.19, [cal[n][1] for n in names], height=0.36, color=[cm[n] for n in names], label="After CQR calibration")
    ax.axvline(0.9, color=INK, ls="--", lw=1); ax.text(0.9, -0.9, " nominal 90%", fontsize=7, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.5); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set(xlabel="Empirical coverage of the 90% interval (test)", xlim=(0.5, 1.0), title="Interval calibration"); ax.legend(fontsize=7, loc="lower left")
    save(fig, outdir, "fig08_calibration")


def fig_coverage_width(rows, outdir):
    fig, ax = plt.subplots(figsize=(4.8, 3.4)); cm = colors([r[0] for r in rows])
    for n, picp, mpiw in rows: ax.scatter(mpiw, picp, s=70 if n.startswith(PROPOSED) else 35, color=cm[n], zorder=3, edgecolor="white"); ax.annotate(n, (mpiw, picp), fontsize=6.5, xytext=(4, 3), textcoords="offset points")
    ax.axhline(0.9, color=MUTED, ls="--", lw=1); ax.set(xlabel="Mean interval width (VWC)", ylabel="Coverage (PICP)", title="Interval quality at 90% nominal (calibrated)")
    save(fig, outdir, "fig09_coverage_vs_width")


# ---------------------------------------------------------------- Fig: ablation, robustness, per-group, importance, irrigation
def fig_ablation(rows, outdir):
    """rows: list of (name, mean, std)."""
    fig, ax = plt.subplots(figsize=(5.4, 2.9)); names = [r[0] for r in rows]
    ax.barh(range(len(rows)), [r[1] for r in rows], xerr=[r[2] for r in rows], color=[VERM if n == PROPOSED else OI[5] for n in names], height=0.6, capsize=2, ecolor=MUTED)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(names); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    for k, r in enumerate(rows): ax.text(r[1], k, f" {r[1]:.4f}", va="center", fontsize=7)
    ax.set(xlabel="Mean MAE over horizons (VWC)", title="Ablation study"); save(fig, outdir, "fig10_ablation")


def fig_robustness(res, outdir):
    """res[kind][model] = (levels, mae list)."""
    kinds = list(res); fig, ax = plt.subplots(1, len(kinds), figsize=(4.6 * len(kinds), 3.3)); ax = np.atleast_1d(ax)
    cm = colors(list(res[kinds[0]]))
    for a, k in zip(ax, kinds):
        for n, (lv, m) in res[k].items(): a.plot(lv, m, color=cm[n], lw=2.4 if n.startswith(PROPOSED) else 1.1, marker="o", ms=3, label=n)
        a.set(xlabel={"noise": "Gaussian noise on input VWC (std units)", "dropout": "Fraction of stuck-at-last-value readings"}[k], ylabel="MAE (VWC)", title={"noise": "Sensor noise", "dropout": "Sensor dropout / stuck values"}[k])
    ax[-1].legend(fontsize=6.3, ncol=2); save(fig, outdir, "fig11_robustness")


def fig_group_heatmap(mat, group_names, model_names, outdir):
    fig, ax = plt.subplots(figsize=(0.55 * len(group_names) + 3, 0.38 * len(model_names) + 1.6))
    im = ax.imshow(mat, aspect="auto", cmap="Blues"); ax.grid(False)
    ax.set_xticks(range(len(group_names))); ax.set_xticklabels([str(g)[:10] for g in group_names], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(model_names))); ax.set_yticklabels(model_names, fontsize=7.5)
    fig.colorbar(im, ax=ax, label="MAE (VWC)", shrink=0.8); ax.set_title("Per-sensor test MAE (mid horizon)")
    save(fig, outdir, "fig12_per_sensor_heatmap")


def fig_importance(names, vals, outdir):
    o = np.argsort(vals); fig, ax = plt.subplots(figsize=(4.8, 0.3 * len(names) + 1.2))
    ax.barh(range(len(o)), np.array(vals)[o], color=OI[0], height=0.6); ax.set_yticks(range(len(o))); ax.set_yticklabels(np.array(names)[o]); ax.grid(axis="y", visible=False)
    ax.set(xlabel="Increase in MAE when feature is shuffled", title="Permutation importance (SoilCastNet)"); save(fig, outdir, "fig13_feature_importance")


def fig_irrigation(curves, outdir):
    """curves: dict model -> (fpr, tpr, auc, prec, rec, ap)"""
    cm = colors(list(curves)); fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
    for n, (fpr, tpr, auc, pr, rc, ap) in curves.items():
        lw = 2.4 if n.startswith(PROPOSED) else 1.0
        ax[0].plot(fpr, tpr, color=cm[n], lw=lw, label=f"{n} ({auc:.3f})"); ax[1].plot(rc, pr, color=cm[n], lw=lw, label=f"{n} ({ap:.3f})")
    ax[0].plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1); ax[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC: soil-moisture deficit event")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall"); ax[0].legend(fontsize=6, loc="lower right"); ax[1].legend(fontsize=6, loc="lower left")
    save(fig, outdir, "fig14_irrigation_trigger")
