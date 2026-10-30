import numpy as np
from scipy import stats


def point_metrics(y, p, y_persist=None):
    e = p - y
    sst = ((y - y.mean()) ** 2).sum() + 1e-12
    nse = 1 - (e ** 2).sum() / sst                                   # Nash-Sutcliffe (hydrology standard)
    r = np.corrcoef(y, p)[0, 1] if y.std() > 0 and p.std() > 0 else 0.0
    kge = 1 - np.sqrt((r - 1) ** 2 + (p.std() / (y.std() + 1e-12) - 1) ** 2 + (p.mean() / (y.mean() + 1e-12) - 1) ** 2)
    out = dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), NSE=nse, KGE=kge, R=r)
    if y_persist is not None:
        mp = np.abs(y_persist - y).mean()
        out["Skill_vs_persist"] = 1 - out["MAE"] / (mp + 1e-12)
    return out


def interval_metrics(y, lo, hi, alpha=0.1):
    inside = (y >= lo) & (y <= hi)
    w = hi - lo
    winkler = w + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)
    return dict(PICP=inside.mean(), MPIW=w.mean(), Winkler=winkler.mean())


def cqr_adjust(y_val, lo_val, hi_val, alpha=0.1):
    """Conformalised quantile regression (Romano et al. 2019) - one correction per horizon column."""
    s = np.maximum(lo_val - y_val, y_val - hi_val)
    n = len(s); lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(s, lvl)


def wilcoxon_abs(err_a, err_b, max_n=20000, seed=0):
    """Paired Wilcoxon signed-rank on absolute errors (a = proposed, b = baseline)."""
    ea, eb = np.abs(err_a), np.abs(err_b)
    if len(ea) > max_n:
        i = np.random.default_rng(seed).choice(len(ea), max_n, replace=False); ea, eb = ea[i], eb[i]
    try: return stats.wilcoxon(ea, eb).pvalue
    except ValueError: return 1.0


def cls_metrics(event, score, pred):
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
    if event.sum() == 0 or event.sum() == len(event): return dict(AUC=np.nan, AP=np.nan, Precision=np.nan, Recall=np.nan, F1=np.nan)
    p, r, f, _ = precision_recall_fscore_support(event, pred, average="binary", zero_division=0)
    return dict(AUC=roc_auc_score(event, score), AP=average_precision_score(event, score), Precision=p, Recall=r, F1=f)
