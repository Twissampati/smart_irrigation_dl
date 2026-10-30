"""Data loading, cleaning, windowing and splitting for soil-moisture forecasting.

Works with ANY real sensor dataset (Kaggle CSV file, or a folder of CSV/TXT files,
one per sensor) as long as you tell it which column is the timestamp and which is
the soil-moisture target. See README.md.
"""
import os, glob
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- loading
def _read_any(path):
    # sep=None lets pandas sniff comma / tab / semicolon / whitespace
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        return pd.read_csv(path, sep=r"\s+", engine="python")


def load_raw(path, group_col=None, path_contains=None):
    """Return one DataFrame; adds a `__group__` column identifying the sensor/farm."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
                       + glob.glob(os.path.join(path, "**", "*.txt"), recursive=True))
        if path_contains:   # e.g. "Hourly" keeps only files whose path contains that text
            files = [f for f in files if path_contains.lower() in f.lower()]
        if not files:
            raise FileNotFoundError(f"No csv/txt files under {path}")
        frames = []
        for f in files:
            d = _read_any(f)
            d["__group__"] = os.path.splitext(os.path.basename(f))[0]
            frames.append(d)
        df = pd.concat(frames, ignore_index=True)
    else:
        df = _read_any(path)
        df["__group__"] = df[group_col].astype(str) if group_col else "site0"
    df.columns = [str(c).strip() for c in df.columns]
    return df


def inspect(path, group_col=None, path_contains=None):
    """Print columns/dtypes/ranges so you can fill in --time-col / --target-col."""
    df = load_raw(path, group_col, path_contains)
    print(df.head(), "\n")
    print(df.dtypes, "\n")
    print(df.describe(include="all").T.iloc[:, :6])
    print("\nGroups:", df["__group__"].nunique(), df["__group__"].unique()[:10])


def prepare_groups(df, time_col, target_col, feature_cols=None, freq="h",
                   max_gap=6, min_len=500, target_range=None, clock_col=None):
    """Clean -> resample -> short-gap interpolate -> engineer features.

    Returns dict group -> DataFrame (index=time, first column = target)."""
    df = df.copy()
    if clock_col:   # separate Date + Time columns -> one timestamp
        df[time_col] = df[time_col].astype(str) + " " + df[clock_col].astype(str)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    if feature_cols is None:  # all other numeric columns
        feature_cols = [c for c in df.columns
                        if c not in (time_col, target_col, "__group__")
                        and pd.api.types.is_numeric_dtype(df[c])]
    cols = [target_col] + list(feature_cols)
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if target_range is not None:  # physically impossible readings -> NaN
        lo, hi = target_range
        df.loc[(df[target_col] < lo) | (df[target_col] > hi), target_col] = np.nan

    out = {}
    for g, d in df.groupby("__group__"):
        d = d.set_index(time_col).sort_index()[cols]
        d = d[~d.index.duplicated()]
        d = d.resample(freq).mean()
        d = d.interpolate(limit=max_gap, limit_area="inside")   # only short gaps
        if len(feature_cols):
            d[feature_cols] = d[feature_cols].ffill(limit=max_gap)
            d[feature_cols] = d[feature_cols].fillna(d[feature_cols].median()).fillna(0.0)
        if d[target_col].notna().sum() < min_len:
            continue
        d = d.rename(columns={target_col: "target"})
        d["d_target"] = d["target"].diff().fillna(0.0)       # first difference (wetting/drying rate)
        idx = d.index
        if freq.lower().startswith("h"):
            d["hr_sin"] = np.sin(2 * np.pi * idx.hour / 24); d["hr_cos"] = np.cos(2 * np.pi * idx.hour / 24)
        d["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25); d["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
        out[g] = d
    if not out:
        raise ValueError("No usable group left after cleaning - check --target-col/--time-col.")
    return out


# ----------------------------------------------------------------------------- windowing
class WindowSet:
    """Vectorised sliding-window access into one big array (no window copies in RAM)."""
    def __init__(self, A, anchors, gid, L, horizons, mean, std):
        self.A, self.anchors, self.gid = A, anchors, gid
        self.L, self.H = L, np.asarray(horizons)
        self.mean, self.std = mean, std      # for target channel (col 0) de-normalisation
        self._off = np.arange(-L + 1, 1)

    def __len__(self): return len(self.anchors)

    def batch(self, idx):
        a = self.anchors[idx]
        X = self.A[a[:, None] + self._off[None, :]]            # (B,L,F)
        Y = self.A[a[:, None] + self.H[None, :], 0]            # (B,H)  normalised target
        return X.astype(np.float32), Y.astype(np.float32)

    def all(self, max_n=None, seed=0):
        idx = np.arange(len(self))
        if max_n and len(idx) > max_n:
            idx = np.sort(np.random.default_rng(seed).choice(idx, max_n, replace=False))
        X, Y = self.batch(idx)
        return X, Y, self.gid[idx]

    def subsample(self, max_n, seed=0):
        if max_n is None or len(self) <= max_n: return self
        keep = np.sort(np.random.default_rng(seed).choice(len(self), max_n, replace=False))
        return WindowSet(self.A, self.anchors[keep], self.gid[keep], self.L, self.H, self.mean, self.std)

    def denorm(self, y): return y * self.std + self.mean


def build_splits(groups, L, horizons, split="temporal", ratios=(0.7, 0.15, 0.15),
                 stride=1, test_group_frac=0.2, seed=0):
    """Chronological split per sensor (no target leakage across split borders).

    split='temporal': every sensor 70/15/15 in time.
    split='spatial' : ~20% of sensors are unseen in training; they are tested on
                      their test period only (spatio-temporal hold-out)."""
    Hmax = int(max(horizons))
    names = sorted(groups)
    rng = np.random.default_rng(seed)
    test_groups = set()
    if split == "spatial":
        k = max(1, int(round(test_group_frac * len(names))))
        test_groups = set(rng.choice(names, k, replace=False))

    # concatenate all sensors in one array with offsets
    arrays, offsets, o = [], {}, 0
    for g in names:
        arrays.append(groups[g].values.astype(np.float64)); offsets[g] = o; o += len(groups[g])
    A = np.concatenate(arrays, 0)
    feat_names = list(groups[names[0]].columns)

    # anchors per split (target not NaN across the whole window and horizon)
    anchors = {"train": [], "val": [], "test": []}; gids = {"train": [], "val": [], "test": []}
    train_rows = []
    for gi, g in enumerate(names):
        T = len(groups[g]); off = offsets[g]
        ok = ~np.isnan(groups[g]["target"].values)
        bad = np.cumsum(~ok)
        def clean(i):  # window [i-L+1, i+Hmax] free of NaN targets
            lo, hi = i - L + 1, i + Hmax
            return (bad[hi] - (bad[lo - 1] if lo > 0 else 0)) == 0
        b1, b2 = int(ratios[0] * T), int((ratios[0] + ratios[1]) * T)
        spans = {"train": (L - 1, b1 - Hmax - 1), "val": (b1, b2 - Hmax - 1), "test": (b2, T - Hmax - 1)}
        for s, (lo, hi) in spans.items():
            if split == "spatial":
                if g in test_groups and s != "test": continue
                if g not in test_groups and s == "test": continue
            ii = np.arange(max(lo, L - 1), hi + 1, stride if s == "train" else 1)
            if len(ii):
                lo_i, hi_i = ii - L + 1, ii + Hmax
                prev = np.where(lo_i > 0, bad[np.maximum(lo_i - 1, 0)], 0)
                ii = ii[(bad[hi_i] - prev) == 0]
            anchors[s].append(ii + off); gids[s].append(np.full(len(ii), gi))
        if g not in test_groups:
            tr = np.arange(0, b1)
            train_rows.append(A[off + tr[ok[tr]]][:, :])
    train_rows = np.concatenate(train_rows, 0)
    mean, std = train_rows.mean(0), train_rows.std(0) + 1e-8
    An = (A - mean) / std
    sets = {s: WindowSet(An, np.concatenate(anchors[s]).astype(np.int64),
                         np.concatenate(gids[s]).astype(np.int64), L, horizons, mean[0], std[0])
            for s in anchors}
    return sets, names, feat_names, dict(mean=mean, std=std)


# ----------------------------------------------------------------------------- stand-in data
def make_demo_csv(path, n_sensors=8, days=420, seed=0):
    """SYNTHETIC stand-in with a Kaggle-like schema, ONLY to smoke-test the pipeline.
    Never report results from this file as 'real-world'."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2020-01-01", periods=days * 24, freq="h")
    rain = (rng.random(len(t)) < 0.012) * rng.gamma(2.0, 6.0, len(t))   # shared weather
    rows = []
    for s in range(n_sensors):
        k = rng.uniform(0.004, 0.012); cap = rng.uniform(0.32, 0.45); th = np.empty(len(t)); th[0] = 0.25
        for i in range(1, len(t)):
            doy = t[i].dayofyear; et = 0.0008 * (1 + 0.6 * np.sin(2 * np.pi * (doy - 80) / 365)) * (0.5 + 0.5 * np.sin(2 * np.pi * (t[i].hour - 6) / 24).clip(0))
            th[i] = np.clip(th[i - 1] + 0.006 * rain[i] - k * (th[i - 1] - 0.08) * 0.3 - et, 0.05, cap)
        temp = 12 + 10 * np.sin(2 * np.pi * (t.dayofyear - 110) / 365) + 3 * np.sin(2 * np.pi * (t.hour - 9) / 24) + rng.normal(0, .5, len(t))
        obs = th + rng.normal(0, 0.003, len(t))
        obs[rng.random(len(t)) < 0.01] = np.nan
        rows.append(pd.DataFrame({"timestamp": t, "sensor_id": f"S{s:02d}", "vwc": obs, "soil_temp": temp}))
    pd.concat(rows).to_csv(path, index=False)
    return path
