"""End-to-end experiment runner: loads data, trains/evaluates baselines + SoilCastNet,
ablations, calibration, robustness, irrigation-trigger proxy, statistics, figures and tables.

Examples
  python run_experiments.py --inspect --data data/soil.csv
  python run_experiments.py --data data/soil.csv --time-col timestamp --target-col vwc --group-col sensor_id
  python run_experiments.py --demo --quick        # smoke test on SYNTHETIC data (not for the paper)
"""
import os, json, argparse, time
import numpy as np, pandas as pd
import torch
import data as D, models as M, metrics as Mx, plots as P
from train import fit, predict, persistence, ridge_fit_predict, perturb, set_seed, DEV

NEURAL = ["MLP", "LSTM", "GRU", "CNN1D", "CNN-LSTM", "TCN", "Transformer", "DLinear", "SoilCastNet"]
ABLATE = ["SoilCastNet w/o multiscale", "SoilCastNet w/o physics", "SoilCastNet w/o gate"]
PROP = "SoilCastNet"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data"); p.add_argument("--time-col"); p.add_argument("--target-col"); p.add_argument("--group-col")
    p.add_argument("--feature-cols", default=None, help="comma list; default = every other numeric column")
    p.add_argument("--clock-col", default=None, help="separate time-of-day column (e.g. Time) when --time-col only holds the date")
    p.add_argument("--path-contains", default=None, help="folder mode: only use files whose path contains this text (e.g. Hourly)")
    p.add_argument("--freq", default="h"); p.add_argument("--max-gap", type=int, default=6)
    p.add_argument("--target-range", default=None, help="e.g. 0,0.7 : readings outside are treated as sensor errors")
    p.add_argument("--lookback", type=int, default=48); p.add_argument("--horizons", default="1,6,24")
    p.add_argument("--split", choices=["temporal", "spatial"], default="temporal")
    p.add_argument("--epochs", type=int, default=30); p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch", type=int, default=512); p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seeds", type=int, default=3); p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max-train", type=int, default=200000); p.add_argument("--max-val", type=int, default=40000); p.add_argument("--max-test", type=int, default=80000)
    p.add_argument("--models", default=",".join(NEURAL)); p.add_argument("--no-ablation", action="store_true"); p.add_argument("--no-robustness", action="store_true")
    p.add_argument("--out", default="results"); p.add_argument("--inspect", action="store_true")
    p.add_argument("--demo", action="store_true", help="synthetic stand-in data, ONLY to test the pipeline")
    p.add_argument("--quick", action="store_true", help="tiny settings for a fast smoke test")
    return p.parse_args()


def pm(m, s): return f"{m:.4f} ± {s:.4f}"


def main():
    a = parse(); os.makedirs(a.out, exist_ok=True); fig_dir = os.path.join(a.out, "figures"); tab_dir = os.path.join(a.out, "tables")
    os.makedirs(tab_dir, exist_ok=True); P.style()
    if a.demo:
        a.data = D.make_demo_csv(os.path.join(a.out, "DEMO_synthetic.csv")); a.time_col, a.target_col, a.group_col = "timestamp", "vwc", "sensor_id"
        print("!! DEMO MODE: synthetic data - for pipeline testing only !!")
    if a.inspect: return D.inspect(a.data, a.group_col, a.path_contains)
    if a.quick: a.epochs, a.seeds, a.max_train, a.max_val, a.max_test = 4, 2, 20000, 8000, 15000
    if not (a.data and a.time_col and a.target_col): raise SystemExit("Need --data, --time-col, --target-col (or --demo). Use --inspect first.")
    horizons = [int(h) for h in a.horizons.split(",")]; H = len(horizons); mid = H // 2
    cfg = dict(L=a.lookback, horizons=horizons, epochs=a.epochs, patience=a.patience, batch=a.batch, lr=a.lr, max_val=a.max_val)
    json.dump({**vars(a), "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))}, open(os.path.join(a.out, "config.json"), "w"), indent=2)

    # ------------------------------------------------------------ data
    raw = D.load_raw(a.data, a.group_col, a.path_contains)
    tr_range = tuple(float(x) for x in a.target_range.split(",")) if a.target_range else None
    feats = a.feature_cols.split(",") if a.feature_cols else None
    groups = D.prepare_groups(raw, a.time_col, a.target_col, feats, a.freq, a.max_gap, target_range=tr_range, clock_col=a.clock_col)
    sets, gnames, feat_names, st = D.build_splits(groups, a.lookback, horizons, a.split, stride=a.stride)
    print(f"{len(groups)} sensors | features: {feat_names} | windows train/val/test = {len(sets['train'])}/{len(sets['val'])}/{len(sets['test'])}")
    test_full = sets["test"]; sets["train"] = sets["train"].subsample(a.max_train); sets["test"] = test_full.subsample(a.max_test)
    P.fig_dataset(groups, fig_dir); P.fig_architecture(fig_dir)

    Xte, Yte_n, gte = sets["test"].all(); Xva, Yva_n, _ = sets["val"].all(max_n=a.max_val)
    dn = sets["test"].denorm; Yte = dn(Yte_n); Yva = dn(Yva_n)
    anchor_te = dn(Xte[:, -1, 0]); thr_amp = np.abs(Yte - anchor_te[:, None])
    event_mask = thr_amp >= np.quantile(thr_amp, 0.9, axis=0, keepdims=True)           # top-10% largest changes per horizon

    store, hist, models, rows, irows = {}, {}, {}, [], []

    def evaluate(name, seed, ptest, pval=None):
        med = ptest[..., 1]
        for j, h in enumerate(horizons):
            r = dict(model=name, seed=seed, horizon=h, **Mx.point_metrics(Yte[:, j], med[:, j], anchor_te))
            r["MAE_event"] = np.abs(med[:, j] - Yte[:, j])[event_mask[:, j]].mean()
            if ptest.shape[-1] == 3 and not np.allclose(ptest[..., 0], ptest[..., 1]):
                raw_i = Mx.interval_metrics(Yte[:, j], ptest[:, j, 0], ptest[:, j, 2]); r.update({f"raw_{k}": v for k, v in raw_i.items()})
                if pval is not None:
                    qh = Mx.cqr_adjust(Yva[:, j], pval[:, j, 0], pval[:, j, 2])
                    ci = Mx.interval_metrics(Yte[:, j], ptest[:, j, 0] - qh, ptest[:, j, 2] + qh); r.update({f"cqr_{k}": v for k, v in ci.items()})
            rows.append(r)

    # ------------------------------------------------------------ non-neural baselines
    pp = np.repeat(anchor_te[:, None], H, 1); store["Persistence"] = {0: np.stack([pp] * 3, -1)}; evaluate("Persistence", 0, store["Persistence"][0])
    rr = ridge_fit_predict(sets["train"], [Xte])[0]; rr = dn(rr); store["Ridge (linear)"] = {0: np.stack([rr] * 3, -1)}; evaluate("Ridge (linear)", 0, store["Ridge (linear)"][0])

    # ------------------------------------------------------------ neural models
    todo = [m for m in a.models.split(",")] + ([] if a.no_ablation else ABLATE)
    t_start = time.time()
    for name in todo:
        store[name] = {}; hist[name] = []
        for seed in range(a.seeds):
            cf = os.path.join(a.out, "cache", f"{name.replace(' ', '_').replace('/', '')}_s{seed}.pt")   # resume support
            if os.path.exists(cf):
                ck = torch.load(cf, map_location=DEV, weights_only=False)
                model = M.make_model(name, a.lookback, sets["train"].A.shape[1], H).to(DEV); model.load_state_dict(ck["state"]); h = ck["hist"]
                print(f"  [{name} s{seed}] loaded from cache (skipping training)", flush=True)
            else:
                model, h = fit(name, sets, cfg, seed)
                os.makedirs(os.path.dirname(cf), exist_ok=True); torch.save({"state": model.state_dict(), "hist": h}, cf)
            hist[name].append(h)
            pt, gt = predict(model, None, Xte); pv, _ = predict(model, None, Xva)
            pt, pv = dn(pt), dn(pv)
            store[name][seed] = pt; evaluate(name, seed, pt, pv)
            if seed == 0:
                models[name] = model
                if name == PROP: gate_te = gt
        print(f"== {name} done ({(time.time() - t_start) / 60:.1f} min elapsed)", flush=True)

    res = pd.DataFrame(rows); res.to_csv(os.path.join(tab_dir, "all_results_per_seed.csv"), index=False)
    agg = res.groupby(["model", "horizon"]).agg(["mean", "std"]).fillna(0.0)
    order = ["Persistence", "Ridge (linear)"] + [m for m in a.models.split(",")]
    # ---- main tables (CSV + LaTeX)
    for metric in ["MAE", "RMSE", "NSE", "KGE", "MAE_event", "Skill_vs_persist"]:
        t = pd.DataFrame({f"+{h}": {m: pm(agg.loc[(m, h), (metric, "mean")], agg.loc[(m, h), (metric, "std")]) for m in order if (m, h) in agg.index} for h in horizons})
        t.to_csv(os.path.join(tab_dir, f"table_{metric}.csv")); open(os.path.join(tab_dir, f"table_{metric}.tex"), "w").write(t.to_latex(escape=True))
    # ---- interval table
    ic = [c for c in agg.columns.levels[0] if c.startswith(("raw_", "cqr_"))]
    if ic: 
        agg[[(c, "mean") for c in ic]].droplevel(1, axis=1).to_csv(os.path.join(tab_dir, "table_intervals.csv"))
        open(os.path.join(tab_dir, "table_intervals.tex"), "w").write(agg[[(c, "mean") for c in ic]].droplevel(1, axis=1).to_latex(float_format="%.4f", escape=True))

    # ---- figures: comparison
    sm = lambda metric, names: {n: [(agg.loc[(n, h), (metric, "mean")], agg.loc[(n, h), (metric, "std")]) for h in horizons] for n in names}
    show = ["Persistence", "Ridge (linear)"] + [m for m in a.models.split(",")]
    P.fig_compare(sm("MAE", show), horizons, fig_dir, "MAE"); P.fig_compare(sm("NSE", show), horizons, fig_dir, "NSE")
    P.fig_metric_vs_horizon(sm("MAE", show), horizons, fig_dir, "MAE"); P.fig_metric_vs_horizon(sm("MAE_event", show), horizons, fig_dir, "MAE_event")
    P.fig_training({k: v for k, v in hist.items() if k in a.models.split(",")}, fig_dir)
    P.fig_scatter(Yte[:, mid], {n: store[n][0][:, mid, 1] for n in show}, fig_dir, horizons[mid])

    # ---- calibration figs
    if "cqr_PICP" in res:
        sub = res.dropna(subset=["cqr_PICP"]).groupby("model")[["raw_PICP", "cqr_PICP"]].mean()
        names_i = [n for n in a.models.split(",") if n in sub.index]
        P.fig_calibration({n: (sub.loc[n, "raw_PICP"], sub.loc[n, "cqr_PICP"]) for n in names_i}, fig_dir)
        s2 = res.dropna(subset=["cqr_PICP"]).groupby("model")[["cqr_PICP", "cqr_MPIW"]].mean()
        P.fig_coverage_width([(n, s2.loc[n, "cqr_PICP"], s2.loc[n, "cqr_MPIW"]) for n in names_i], fig_dir)

    # ---- ablation
    if not a.no_ablation:
        ab = res[res.model.isin([PROP] + ABLATE)].groupby(["model", "seed"]).MAE.mean().groupby("model").agg(["mean", "std"]).fillna(0)
        P.fig_ablation([(n, ab.loc[n, "mean"], ab.loc[n, "std"]) for n in [PROP] + ABLATE], fig_dir)
        ab.to_csv(os.path.join(tab_dir, "table_ablation.csv"))

    # ---- statistical significance (paired Wilcoxon on |error|, seed 0)
    sig = []
    for n in show:
        if n == PROP: continue
        for j, h in enumerate(horizons):
            sig.append(dict(baseline=n, horizon=h, MAE_baseline=np.abs(store[n][0][:, j, 1] - Yte[:, j]).mean(), MAE_proposed=np.abs(store[PROP][0][:, j, 1] - Yte[:, j]).mean(),
                            p_value=Mx.wilcoxon_abs(store[PROP][0][:, j, 1] - Yte[:, j], store[n][0][:, j, 1] - Yte[:, j])))
    pd.DataFrame(sig).to_csv(os.path.join(tab_dir, "table_significance.csv"), index=False)

    # ---- forecast time-series figure (consecutive windows of the first test sensor)
    g0 = np.bincount(test_full.gid).argmax(); idx = np.where(test_full.gid == g0)[0][:600]
    Xs, Ys_n = test_full.batch(idx); ps, gs = predict(models[PROP], None, Xs); ps = dn(ps); Ys = dn(Ys_n)
    qh = Mx.cqr_adjust(Yva[:, mid], dn(predict(models[PROP], None, Xva)[0])[:, mid, 0], dn(predict(models[PROP], None, Xva)[0])[:, mid, 2])
    ev = ((Ys_n[:, mid] - Xs[:, -1, 0]) > 0.05).astype(float)
    P.fig_forecast_ts(None, Ys[:, mid], ps[:, mid, 1], ps[:, mid, 0] - qh, ps[:, mid, 2] + qh, gs[:, mid], ev, fig_dir, f"{horizons[mid]} step(s)")

    # ---- per-sensor heatmap
    ug = np.unique(gte); ug = ug[np.argsort([-np.sum(gte == g) for g in ug])][:25]
    hm_models = show[2:] if len(show) > 2 else show
    mat = np.array([[np.abs(store[n][0][gte == g, mid, 1] - Yte[gte == g, mid]).mean() for g in ug] for n in hm_models])
    P.fig_group_heatmap(mat, [gnames[g] for g in ug], hm_models, fig_dir)

    # ---- permutation importance (proposed)
    sel = np.random.default_rng(0).choice(len(Xte), min(10000, len(Xte)), replace=False); Xs_ = Xte[sel]
    base = np.abs(dn(predict(models[PROP], None, Xs_)[0])[:, mid, 1] - Yte[sel, mid]).mean(); imp = []
    for j in range(Xs_.shape[2]):
        Xp = Xs_.copy(); Xp[:, :, j] = Xp[np.random.default_rng(j).permutation(len(Xp))][:, :, j]
        imp.append(np.abs(dn(predict(models[PROP], None, Xp)[0])[:, mid, 1] - Yte[sel, mid]).mean() - base)
    P.fig_importance(feat_names, imp, fig_dir); pd.DataFrame(dict(feature=feat_names, delta_MAE=imp)).to_csv(os.path.join(tab_dir, "table_feature_importance.csv"), index=False)

    # ---- robustness
    if not a.no_robustness:
        rn = [n for n in ["LSTM", "GRU", "CNN-LSTM", "TCN", "Transformer", PROP] if n in models]; sub = np.random.default_rng(1).choice(len(Xte), min(20000, len(Xte)), replace=False)
        grid = {"noise": [0, 0.05, 0.1, 0.2, 0.4], "dropout": [0, 0.1, 0.2, 0.4]}; rob = {}; rrows = []
        for kind, lv in grid.items():
            rob[kind] = {}
            for n in rn + ["Persistence"]:
                m = []
                for l in lv:
                    Xp = perturb(Xte[sub], kind, l, st, feat_names)
                    pr = dn(Xp[:, -1, 0]) if n == "Persistence" else dn(predict(models[n], None, Xp)[0])[:, mid, 1]
                    m.append(np.abs(pr - Yte[sub, mid]).mean()); rrows.append(dict(kind=kind, level=l, model=n, MAE=m[-1]))
                rob[kind][n] = (lv, m)
        P.fig_robustness(rob, fig_dir); pd.DataFrame(rrows).to_csv(os.path.join(tab_dir, "table_robustness.csv"), index=False)

    # ---- irrigation-trigger PROXY (no real irrigation labels exist in soil-moisture-only datasets!)
    thr_g = np.array([np.nanquantile(groups[g]["target"].values[:int(0.7 * len(groups[g]))], 0.2) for g in gnames])
    thr = thr_g[gte]; ev = Yte[:, mid] < thr; irows = []; curves = {}
    print(f"Irrigation-proxy event prevalence in test set: {ev.mean():.3f}" + ("  (no events -> classification metrics are NaN; raise the percentile)" if ev.mean() == 0 else ""))
    from sklearn.metrics import roc_curve, precision_recall_curve
    for n in show:
        pt = store[n][0]; score = thr - pt[:, mid, 1]; rules = {"median < thr": pt[:, mid, 1] < thr}
        if n in res.dropna(subset=["cqr_PICP"]).model.unique():
            pv = dn(predict(models[n], None, Xva)[0]); qh = Mx.cqr_adjust(Yva[:, mid], pv[:, mid, 0], pv[:, mid, 2]); rules["risk-averse (lower bound < thr)"] = (pt[:, mid, 0] - qh) < thr
        for rule, pred in rules.items(): irows.append(dict(model=n, rule=rule, **Mx.cls_metrics(ev, score, pred)))
        if ev.sum() and not ev.all():
            f, t_, _ = roc_curve(ev, score); pr_, rc_, _ = precision_recall_curve(ev, score); curves[n] = (f, t_, Mx.cls_metrics(ev, score, rules["median < thr"])["AUC"], pr_, rc_, Mx.cls_metrics(ev, score, rules["median < thr"])["AP"])
    pd.DataFrame(irows).to_csv(os.path.join(tab_dir, "table_irrigation_trigger_proxy.csv"), index=False)
    if curves: P.fig_irrigation({k: v for k, v in curves.items() if k in [PROP, "LSTM", "GRU", "CNN-LSTM", "Transformer", "Persistence"]}, fig_dir)

    print(f"\nDone. Tables -> {tab_dir}\nFigures -> {fig_dir}")
    print(pd.read_csv(os.path.join(tab_dir, "table_MAE.csv"), index_col=0).to_string())


if __name__ == "__main__":
    main()
