"""Rich per-file features for IMS. Run locally where the IMS folder exists:
   python extract_features.py IMS [--force]
Skips any case whose CSV already exists. Writes case{1,2,3}_features_v2.csv (same row order as the old CSVs)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import kurtosis, skew

FS = 20_000
CASES = {1: ("1st_test/1st_test", 4), 2: ("2nd_test/2nd_test", 0), 3: ("3rd_test/txt", 2)}
FREQ = dict(BPFO=236.4, BPFI=296.9, BSF=139.9)            # Rexnord ZA-2115 @ 2000 rpm
BANDS = [(10, 1000), (1000, 3000), (3000, 6000), (6000, 9500)]
SOS_BP = butter(4, [2000, 9000], "band", fs=FS, output="sos")   # demodulation band

def feats(x):
    x = x - x.mean(); r = np.sqrt(np.mean(x ** 2)); f = {"RMS": r, "Peak": np.abs(x).max(), "Kurtosis": kurtosis(x, fisher=False),
         "Skew": skew(x), "Crest": np.abs(x).max() / r}
    sp = np.abs(np.fft.rfft(x)) / len(x); fr = np.fft.rfftfreq(len(x), 1 / FS)
    for lo, hi in BANDS: f[f"band_{lo}_{hi}"] = np.sqrt(np.sum(sp[(fr >= lo) & (fr < hi)] ** 2) * 2)
    env = np.abs(hilbert(sosfiltfilt(SOS_BP, x))); env -= env.mean()
    es = np.abs(np.fft.rfft(env)) / len(env); ef = np.fft.rfftfreq(len(env), 1 / FS)
    for k, f0 in FREQ.items():
        for h in (1, 2, 3):
            m = (ef > h * f0 - 3) & (ef < h * f0 + 3)
            f[f"env_{k}_h{h}"] = es[m].max()
        f[f"{k}_rms"] = np.sqrt(sum(f[f"env_{k}_h{h}"] ** 2 for h in (1, 2, 3)))
    return f

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--force"]; force = "--force" in sys.argv
    root = Path(args[0])
    for i, (folder, col) in CASES.items():
        out = Path(f"case{i}_features_v2.csv")
        if out.exists() and not force:
            print(f"case{i}: {out} exists, skipping (use --force to redo)"); continue
        files = sorted((root / folder).iterdir())
        pd.DataFrame([feats(np.loadtxt(p)[:, col]) for p in files]).to_csv(out, index=False)
        print(i, len(files))
