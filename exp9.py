import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
df = pd.read_csv("smart_campus_dataset_10K.csv")
# ===================== 1. Bernoulli MLE (hvac_state) =====================
x = df["hvac_state"].values
n = len(x)
p_hat = x.mean() # closed-form MLE = sample proportion
se_p = np.sqrt(p_hat * (1 - p_hat) / n) # from Fisher information
print(f"Bernoulli: p_hat={p_hat:.4f}, SE={se_p:.4f}, "
 f"95% Wald CI=({p_hat - 1.96*se_p:.4f}, {p_hat + 1.96*se_p:.4f})")
# ===================== 2. Normal MLE (indoor_temperature) =====================
y = df["indoor_temperature"].values
mu_hat = y.mean() # analytic MLE of mean
sigma2_hat = y.var(ddof=0) # analytic MLE of variance (divide by n)
print(f"Normal analytic MLE: mu={mu_hat:.4f}, sigma={np.sqrt(sigma2_hat):.4f}")
# Numerical MLE: minimise negative log-likelihood
def neg_loglik_normal(theta, data):
 mu, log_sigma = theta # log-parametrisation keeps sigma > 0
 return -np.sum(stats.norm.logpdf(data, mu, np.exp(log_sigma)))
res = optimize.minimize(neg_loglik_normal, x0=[20, 1], args=(y,), method="Nelder-Mead")
print(f"Normal numerical MLE: mu={res.x[0]:.4f}, sigma={np.exp(res.x[1]):.4f}, "
 f"max loglik={-res.fun:.2f}")
print("scipy norm.fit :", np.round(stats.norm.fit(y), 4))
# Standard errors from the Fisher information
print(f"SE(mu)={np.sqrt(sigma2_hat / len(y)):.4f}, SE(sigma)={np.sqrt(sigma2_hat / (2*len(y))):.4f}")
# Log-likelihood curve for mu (sigma fixed at its MLE)
grid = np.linspace(mu_hat - 0.3, mu_hat + 0.3, 200)
ll = [np.sum(stats.norm.logpdf(y, m, np.sqrt(sigma2_hat))) for m in grid]
plt.figure(figsize=(7, 4.5))
plt.plot(grid, ll)
plt.axvline(mu_hat, color="r", ls="--", label=f"MLE = {mu_hat:.3f}")
plt.xlabel("mu"); plt.ylabel("log-likelihood"); plt.legend()
plt.title("Log-likelihood of mu (indoor_temperature)")
plt.show()
# ===================== 3. Poisson MLE (entry_count) =====================
c = df["entry_count"].values
lam_hat = c.mean() # closed-form MLE of lambda
nll_pois = lambda lam: -np.sum(stats.poisson.logpmf(c, lam[0]))
res_p = optimize.minimize(nll_pois, x0=[100], bounds=[(1e-6, None)])
print(f"Poisson: analytic lambda={lam_hat:.4f}, numeric lambda={res_p.x[0]:.4f}")
# ===================== 4. Choosing a distribution by MLE + AIC / BIC =====================
data = df.loc[df["occupancy_count"] > 0, "occupancy_count"].values.astype(float)
candidates = {"Normal": stats.norm, "Gamma": stats.gamma, "Lognormal": stats.lognorm,
 "Weibull": stats.weibull_min, "Exponential": stats.expon}
rows = []
plt.figure(figsize=(10, 5))
plt.hist(data, bins=40, density=True, alpha=0.4, color="grey")
xs = np.linspace(data.min(), data.max(), 300)
for name, dist in candidates.items():
 params = dist.fit(data) # scipy fits by maximum likelihood
 ll = np.sum(dist.logpdf(data, *params))
 k = len(params)
 ks = stats.kstest(data, dist.cdf, args=params)
 rows.append([name, ll, 2 * k - 2 * ll, k * np.log(len(data)) - 2 * ll, ks.pvalue])
 plt.plot(xs, dist.pdf(xs, *params), label=name)
plt.legend(); plt.title("MLE-fitted distributions for occupancy_count")
plt.show()
fit_tab = pd.DataFrame(rows, columns=["distribution", "loglik", "AIC", "BIC", "KS p-value"])
print(fit_tab.sort_values("AIC").round(3))