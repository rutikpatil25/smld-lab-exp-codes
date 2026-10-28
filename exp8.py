import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
df = pd.read_csv("smart_campus_dataset_10K.csv")
rng = np.random.default_rng(42)
# Draw one realistic sample (n = 300) and pretend it is all we have
sample = df["power_kwh"].sample(300, random_state=1).values
n = len(sample)
B = 10000 # number of bootstrap resamples
print("True population mean:", round(df["power_kwh"].mean(), 4))
# ---------- 1. Classical t-based 95% CI for the mean ----------
se = sample.std(ddof=1) / np.sqrt(n)
t_ci = stats.t.interval(0.95, df=n - 1, loc=sample.mean(), scale=se)
print("t-interval :", np.round(t_ci, 4))
# ---------- 2. Bootstrap resampling (vectorised) ----------
idx = rng.integers(0, n, size=(B, n))
boot = sample[idx] # B x n resampled data
boot_means = boot.mean(axis=1)
boot_medians = np.median(boot, axis=1)
boot_stds = boot.std(axis=1, ddof=1)
# ---------- 3. Percentile bootstrap CI ----------
pct_ci = np.percentile(boot_means, [2.5, 97.5])
print("Percentile CI mean:", np.round(pct_ci, 4))
print("Bootstrap SE :", round(boot_means.std(), 4), "| formula SE:", round(se, 4))
# ---------- 4. Basic (reverse percentile) bootstrap CI ----------
basic_ci = [2 * sample.mean() - pct_ci[1], 2 * sample.mean() - pct_ci[0]]
print("Basic CI mean :", np.round(basic_ci, 4))
# ---------- 5. BCa CI with scipy (bias-corrected & accelerated) ----------
bca = stats.bootstrap((sample,), np.mean, confidence_level=0.95, n_resamples=B,
 method="BCa", random_state=42)
print("BCa CI mean :", np.round([bca.confidence_interval.low,
 bca.confidence_interval.high], 4))
# ---------- 6. Bootstrap CI for statistics with no simple formula ----------
for name, arr in [("median", boot_medians), ("std dev", boot_stds)]:
 print(f"95% bootstrap CI for {name:8s}:", np.round(np.percentile(arr, [2.5, 97.5]), 4))
# Correlation between occupancy_count and capacity (paired resampling of rows)
pairs = df[["occupancy_count", "capacity"]].sample(300, random_state=2).values
ridx = rng.integers(0, 300, size=(B, 300))
boot_r = np.array([np.corrcoef(pairs[i, 0], pairs[i, 1])[0, 1] for i in ridx])
print("Sample r =", round(np.corrcoef(pairs.T)[0, 1], 4),
 "| 95% CI:", np.round(np.percentile(boot_r, [2.5, 97.5]), 4))
# ---------- 7. Visualisation ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(boot_means, bins=50, color="steelblue", edgecolor="white")
axes[0].axvline(pct_ci[0], color="r", ls="--"); axes[0].axvline(pct_ci[1], color="r", ls="--")
axes[0].axvline(df["power_kwh"].mean(), color="k", label="population mean")
axes[0].set_title("Bootstrap distribution of the mean (95% CI)"); axes[0].legend()
axes[1].hist(boot_r, bins=50, color="darkorange", edgecolor="white")
axes[1].axvline(np.percentile(boot_r, 2.5), color="r", ls="--")
axes[1].axvline(np.percentile(boot_r, 97.5), color="r", ls="--")
axes[1].set_title("Bootstrap distribution of correlation r")
plt.tight_layout()
plt.show()
# ---------- 8. Coverage check: does a 95% CI capture the true mean ~95% of the time? ----------
true_mean, hits, trials = df["power_kwh"].mean(), 0, 500
for _ in range(trials):
 s = df["power_kwh"].sample(100).values
 bm = s[rng.integers(0, 100, size=(2000, 100))].mean(axis=1)
 lo, hi = np.percentile(bm, [2.5, 97.5])
 hits += (lo <= true_mean <= hi)
print("Empirical coverage of percentile bootstrap CI:", hits / trials)