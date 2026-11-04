#!/usr/bin/env python
"""IMS bearing CLI.
  python ims_cli.py extract IMS                      # raw files -> case{1,2,3}_features_v2.csv
  python ims_cli.py run all   --data .               # health + fault + rul
  python ims_cli.py run rul   --data . --plots plots # one task, save figures
All evaluation is Leave-One-Set-Out over case1/2/3 (no random splits)."""
import argparse, warnings
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

FEATS = ["RMS", "Peak", "Kurtosis", "Skew", "Crest", "band_10_1000", "band_1000_3000", "band_3000_6000", "band_6000_9500",
         "BPFO_rms", "BPFI_rms", "BSF_rms"]
FN = [f + "_n" for f in FEATS]
FAULT = {"case1": "BPFI", "case2": "BPFO", "case3": "BPFO"}
HEALTHY_FRAC, TRAIN_FRAC, K_IQR, SMOOTH, WIN, SEED, PERSIST, RW = 0.30, 0.20, 3, 5, 10, 0, 5, 50

def load(data):
    def prep(d):
        n = len(d); h = int(HEALTHY_FRAC * n); idx = np.arange(n); b = d["RMS"].iloc[:h]; q1, q3 = b.quantile([.25, .75])
        ex = (d["RMS"].rolling(SMOOTH, min_periods=1).median().values > b.median() + K_IQR * (q3 - q1)) & (idx >= h)
        st = np.maximum.accumulate(ex).astype(int); on = int(st.argmax()) if st.any() else n - 1
        d = d.copy(); d["file_idx"], d["n_files"], d["life"], d["onset_idx"], d["state"] = idx, n, idx / (n - 1), on, st
        d["RUL_percent"] = np.where(idx <= on, 100.0, 100.0 * (n - 1 - idx) / max(n - 1 - on, 1))
        d[FN] = d[FEATS] / d[FEATS].iloc[:int(TRAIN_FRAC * n)].median().values
        return d
    df = pd.concat([prep(pd.read_csv(Path(data) / f"case{i}_features_v2.csv").assign(set_id=f"case{i}")) for i in (1, 2, 3)], ignore_index=True)
    df["fault"] = df["set_id"].map(FAULT)
    return df

def save(fig, plots, name):
    if plots:
        Path(plots).mkdir(exist_ok=True, parents=True); fig.savefig(Path(plots) / name, dpi=130, bbox_inches="tight")
    plt.close(fig)

def health(df, plots):
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler
    print("== Task 1a: Isolation Forest (fit on first 20 % of each set)")
    rows, fig = [], plt.figure(figsize=(11, 8))
    for k, (sid, d) in enumerate(df.groupby("set_id"), 1):
        n, h = len(d), int(TRAIN_FRAC * len(d)); X = StandardScaler().fit(d[FN].iloc[:h]).transform(d[FN])
        s = IsolationForest(n_estimators=300, random_state=SEED).fit(X[:h]).score_samples(X); thr = np.quantile(s[:h], .01)
        run = pd.Series(s < thr).astype(int).rolling(PERSIST).sum().values == PERSIST; run[:h] = False
        al = int(np.argmax(run)) - PERSIST + 1 if run.any() else None; on = int(d["onset_idx"].iloc[0])
        rows.append(dict(set_id=sid, onset=on, alarm=al, lag_files=None if al is None else al - on,
                         lag_pct=None if al is None else round(100 * (al - on) / n, 2)))
        a = fig.add_subplot(3, 1, k); a.plot(s, lw=.6); a.axhline(thr, c="r", ls="--"); a.axvline(on, c="g")
        if al is not None: a.axvline(al, c="k", ls=":")
        a.set_title(sid)
    print(pd.DataFrame(rows).set_index("set_id").to_string()); save(fig, plots, "health_anomaly.png")
    y, g = df["state"].values, df["set_id"].values; no_rms = [f for f in FN if f not in ("RMS_n", "Peak_n", "Crest_n") and not f.startswith("band_")]
    for name, cols in [("with RMS (circular: label derives from RMS)", FN), ("WITHOUT RMS-like features (honest)", no_rms)]:
        pred = np.zeros(len(df), int)
        for tr, te in LeaveOneGroupOut().split(df, y, g):
            pred[te] = RandomForestClassifier(300, class_weight="balanced", n_jobs=-1, random_state=SEED).fit(df.loc[tr, cols], y[tr]).predict(df.loc[te, cols])
        print(f"\n== Task 1b: Random Forest LOSO, {name}\n", confusion_matrix(y, pred), "\n", classification_report(y, pred, target_names=["healthy", "degraded"], digits=3))

def fault(df, plots):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import LeaveOneGroupOut
    h0 = df["file_idx"] < TRAIN_FRAC * df["n_files"]
    sm = df.groupby("set_id")[["BPFO_rms", "BPFI_rms"]].transform(lambda c: c.rolling(RW, min_periods=1).mean())
    r = sm["BPFO_rms"] / sm["BPFI_rms"]; df["ratio_rel"] = r / r.where(h0).groupby(df["set_id"]).transform("median")
    for c, o in (("lb", "BPFO_rms"), ("li", "BPFI_rms")):
        df[c] = np.log(sm[o] / df[o].where(h0).groupby(df["set_id"]).transform("median"))
    dg = df[df["state"] == 1].reset_index(drop=True); dg["rule"] = np.where(dg["ratio_rel"] > 1, "BPFO", "BPFI"); dg["linear"] = None
    for tr, te in LeaveOneGroupOut().split(dg, groups=dg["set_id"]):
        if dg.loc[tr, "fault"].nunique() < 2: continue
        w = 1.0 / dg.loc[tr, "set_id"].map(dg.loc[tr, "set_id"].value_counts())
        m = LogisticRegression(class_weight="balanced", max_iter=1000).fit(dg.loc[tr, ["lb", "li"]], dg.loc[tr, "fault"], sample_weight=w.values)
        dg.loc[te, "linear"] = m.predict(dg.loc[te, ["lb", "li"]])
    print("== Task 2: fault diagnosis on degraded files (linear undefined when case1 is left out: no BPFI in training)")
    print(pd.DataFrame([dict(set_id=s, truth=d["fault"].iloc[0], rule_acc=(d["rule"] == d["fault"]).mean(),
          linear_acc=(d["linear"] == d["fault"]).mean() if d["linear"].notna().any() else np.nan) for s, d in dg.groupby("set_id")]).set_index("set_id").round(3).to_string())
    print("\nRule:\n", classification_report(dg["fault"], dg["rule"], digits=3, zero_division=0))
    fig, ax = plt.subplots(figsize=(10, 4))
    for sid, d in df.groupby("set_id"): ax.semilogy(d["life"], d["ratio_rel"], lw=.8, label=f"{sid} ({FAULT[sid]})")
    ax.axhline(1, c="k", ls="--"); ax.legend(); save(fig, plots, "fault_ratio.png")

def rul(df, plots):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error as MAE
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor
    gb = df.groupby("set_id")[FN]; roll = gb.rolling(WIN, min_periods=1)
    t = np.arange(WIN) - (WIN - 1) / 2
    sl = gb.transform(lambda c: c.rolling(WIN).apply(lambda v: (t * (v - v.mean())).sum() / (t ** 2).sum(), raw=True)).fillna(0).add_suffix("_slope")
    X = pd.concat([df[FN], roll.mean().reset_index(level=0, drop=True).add_suffix("_rmean"),
                   roll.std().reset_index(level=0, drop=True).fillna(0).add_suffix("_rstd"), sl], axis=1).loc[df.index]
    y, g, st = df["RUL_percent"].values, df["set_id"].values, df["state"].values
    models = {"Ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10)),
              "RandomForest": lambda: RandomForestRegressor(300, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
              "XGBoost": lambda: XGBRegressor(n_estimators=400, max_depth=4, learning_rate=.05, subsample=.8, colsample_bytree=.8, random_state=SEED, n_jobs=-1)}
    oof, res = {k: np.zeros(len(df)) for k in models}, []
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        for k, mk in models.items():
            p = oof[k][te] = np.clip(mk().fit(X.iloc[tr], y[tr]).predict(X.iloc[te]), 0, 100); d = st[te] == 1
            res.append(dict(left_out=g[te][0], model=k, MAE_all=MAE(y[te], p), MAE_degraded=MAE(y[te][d], p[d])))
    R = pd.DataFrame(res); print("== Task 3: RUL (MAE in % of life, LOSO)")
    print(R.pivot(index="left_out", columns="model", values="MAE_all").round(2).to_string(), "\n(all files)")
    print(R.pivot(index="left_out", columns="model", values="MAE_degraded").round(2).to_string(), "\n(degraded files only)")
    print("Mean:\n", R.groupby("model")[["MAE_all", "MAE_degraded"]].mean().round(2))
    fig, ax = plt.subplots(3, 1, figsize=(11, 10))
    for a, sid in zip(ax, ["case1", "case2", "case3"]):
        m = g == sid; a.plot(df.loc[m, "life"], y[m], "k", lw=2, label="true")
        for k in models: a.plot(df.loc[m, "life"], oof[k][m], lw=.8, label=k)
        a.set_title(f"left-out {sid}"); a.legend(ncol=4)
    save(fig, plots, "rul_curves.png")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="raw IMS folder -> feature CSVs"); e.add_argument("ims_dir"); e.add_argument("--force", action="store_true", help="re-extract even if CSVs exist")
    r = sub.add_parser("run", help="run ML tasks"); r.add_argument("task", choices=["health", "fault", "rul", "all"])
    r.add_argument("--data", default=".", help="folder with case*_features_v2.csv"); r.add_argument("--plots", default=None, help="save figures to this folder")
    a = ap.parse_args()
    if a.cmd == "extract":
        import runpy, sys
        sys.argv = ["extract_features.py", a.ims_dir] + (["--force"] if a.force else [])
        runpy.run_path(str(Path(__file__).with_name("extract_features.py")), run_name="__main__")
        return
    df = load(a.data)
    for name, fn in (("health", health), ("fault", fault), ("rul", rul)):
        if a.task in (name, "all"): fn(df, a.plots); print()

if __name__ == "__main__":
    main()
