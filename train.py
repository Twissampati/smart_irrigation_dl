import time, copy
import numpy as np
import torch
from models import make_model, pinball, soilcast_loss

torch.set_num_threads(max(1, torch.get_num_threads()))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def _bounds(train_ws, margin=0.25):
    _, Y, _ = train_ws.all(max_n=100000)
    return (float(np.quantile(Y, 0.001)) - margin, float(np.quantile(Y, 0.999)) + margin)


@torch.no_grad()
def predict(model, ws, X=None, bs=2048, max_n=None):
    """Return normalised quantile forecasts (N,H,3) and gate probabilities (N,H) (NaN if n/a)."""
    model.eval()
    if X is None: X, _, _ = ws.all(max_n=max_n)
    out, gate = [], []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(DEV)
        out.append(model(xb).cpu().numpy())
        a = model.aux
        gate.append(a["gate"].cpu().numpy() if (a is not None and a.get("gate") is not None) else np.full(out[-1].shape[:2], np.nan))
    return np.concatenate(out), np.concatenate(gate)


def fit(name, sets, cfg, seed, verbose=True):
    set_seed(seed)
    tr, va = sets["train"], sets["val"]
    L, Fin, H = cfg["L"], tr.A.shape[1], len(cfg["horizons"])
    model = make_model(name, L, Fin, H).to(DEV)
    is_sc = name.startswith("SoilCastNet")
    bounds = _bounds(tr)
    Xv, Yv, _ = va.all(max_n=cfg["max_val"])
    Xv_t, Yv_t = torch.from_numpy(Xv).to(DEV), torch.from_numpy(Yv).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    best, best_state, bad, hist = 1e9, None, 0, []
    n = len(tr); bs = cfg["batch"]
    for ep in range(cfg["epochs"]):
        model.train(); t0 = time.time(); perm = np.random.permutation(n); tl = 0.0; nb = 0
        for i in range(0, n, bs):
            X, Y = tr.batch(perm[i:i + bs])
            xb, yb = torch.from_numpy(X).to(DEV), torch.from_numpy(Y).to(DEV)
            opt.zero_grad()
            pred = model(xb)
            loss = soilcast_loss(model, pred, yb, bounds) if is_sc else pinball(pred, yb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tl += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            vl = pinball(torch.cat([model(Xv_t[j:j + 4096]) for j in range(0, len(Xv_t), 4096)]), Yv_t).item()
        sched.step(vl); hist.append((tl / nb, vl))
        if verbose: print(f"  [{name} s{seed}] ep{ep + 1:02d} train {tl / nb:.4f} val {vl:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if vl < best - 1e-5: best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]: break
    model.load_state_dict(best_state)
    return model, np.array(hist)


# ---------------------------------------------------------------- non-neural baselines
def persistence(X, H):
    return np.repeat(X[:, -1, 0][:, None], H, 1)                # normalised


def ridge_fit_predict(train_ws, X_eval_list, k=24, alpha=1.0, max_n=200000):
    from sklearn.linear_model import Ridge
    X, Y, _ = train_ws.all(max_n=max_n)
    f = lambda X_: X_[:, -k:, :].reshape(len(X_), -1)
    m = Ridge(alpha=alpha).fit(f(X), Y)
    return [m.predict(f(x)) for x in X_eval_list]


# ---------------------------------------------------------------- robustness perturbations
def perturb(X, kind, level, stats, feat_names, seed=0):
    """Simulate sensor problems on the INPUT target history only. kind: 'noise' (std units) | 'dropout' (fraction stuck)."""
    rng = np.random.default_rng(seed); X = X.copy()
    if kind == "noise":
        X[:, :, 0] += rng.normal(0, level, X[:, :, 0].shape)
    elif kind == "dropout":
        m = rng.random(X[:, :, 0].shape) < level
        for t in range(1, X.shape[1]):                      # stuck-at-last-value on masked steps
            X[:, t, 0] = np.where(m[:, t], X[:, t - 1, 0], X[:, t, 0])
    if "d_target" in feat_names:                            # keep derived channel consistent
        j = feat_names.index("d_target"); mu, sd = stats["mean"], stats["std"]
        phys = X[:, :, 0] * sd[0] + mu[0]
        d = np.diff(phys, axis=1, prepend=phys[:, :1])
        X[:, :, j] = (d - mu[j]) / sd[j]
    return X
