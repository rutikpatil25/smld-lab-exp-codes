import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
df = pd.read_csv("smart_campus_dataset_10K.csv")
# ---------- 1. Define target and predictors ----------
df["net_flow"] = df["entry_count"] - df["exit_count"] # engineered (will be collinear)
df["total_flow"] = df["entry_count"] + df["exit_count"] # engineered (will be collinear)
target = "occupancy_count"
features = ["capacity", "idle_minutes", "power_kwh", "hvac_state", "indoor_temperature",
 "outdoor_temperature_index", "cpu_utilization", "power_draw", "runtime_hours",
 "device_temperature", "connected_devices", "bandwidth_usage", "density_index",
 "entry_count", "exit_count", "net_flow", "total_flow"]
# ---------- 2. Multicollinearity: VIF ----------
def compute_vif(data):
    out = {}
    for col in data.columns:
        others = data.drop(columns=col)
        r2 = LinearRegression().fit(others, data[col]).score(others, data[col])
        out[col] = np.inf if r2 > 1 - 1e-10 else 1 / (1 - r2)
    return pd.Series(out, name="VIF").sort_values(ascending=False)
vif = compute_vif(df[features])
print(vif.round(2)) # VIF > 5 (or 10) => serious multicollinearity
plt.figure(figsize=(12, 9))
sns.heatmap(df[features].corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f",
 annot_kws={"size": 6})
plt.title("Predictor correlation matrix")
plt.show()
# Remove redundant predictors until every VIF < 5
selected = features.copy()
while True:
 v = compute_vif(df[selected])
 if v.iloc[0] < 5:
    break
 selected.remove(v.index[0])
print("Dropped:", [f for f in features if f not in selected])
print(compute_vif(df[selected]).round(3))
# ---------- 3. OLS model from scratch (coefficients, SE, t, p, R2, F) ----------
X_train, X_test, y_train, y_test = train_test_split(df[selected], df[target],
 test_size=0.2, random_state=42)
def ols_fit(X, y):
 Xc = np.column_stack([np.ones(len(X)), X.values]) # add intercept
 XtX_inv = np.linalg.inv(Xc.T @ Xc)
 beta = XtX_inv @ Xc.T @ y.values
 fitted = Xc @ beta
 resid = y.values - fitted
 n, p = Xc.shape
 sigma2 = resid @ resid / (n - p) # MSE
 se = np.sqrt(np.diag(sigma2 * XtX_inv))
 tvals = beta / se
 pvals = 2 * (1 - stats.t.cdf(np.abs(tvals), df=n - p))
 sst = ((y.values - y.mean()) ** 2).sum()
 r2 = 1 - resid @ resid / sst
 adj = 1 - (1 - r2) * (n - 1) / (n - p)
 F = (r2 / (p - 1)) / ((1 - r2) / (n - p))
 table = pd.DataFrame({"coef": beta, "std err": se, "t": tvals, "P>|t|": pvals},
 index=["const"] + list(X.columns))
 h = np.einsum("ij,jk,ik->i", Xc, XtX_inv, Xc) # leverage (hat-matrix diagonal)
 return table, fitted, resid, h, sigma2, r2, adj, F, p
table, fitted, resid, h, mse, r2, adj, F, p = ols_fit(X_train, y_train)
print(table.round(4))
print(f"R2={r2:.4f} Adj R2={adj:.4f} F={F:.2f} (p-value="
 f"{1 - stats.f.cdf(F, p - 1, len(y_train) - p):.4g})")
# Test-set performance (scikit-learn)
lr = LinearRegression().fit(X_train, y_train)
pred = lr.predict(X_test)
print("Test R2 :", round(r2_score(y_test, pred), 4))
print("Test RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 4))
# ---------- 4. Residual diagnostics ----------
std_resid = resid / np.sqrt(mse * (1 - h)) # studentised residuals
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0, 0].scatter(fitted, resid, alpha=0.3, s=8)
axes[0, 0].axhline(0, color="r"); axes[0, 0].set_title("Residuals vs fitted")
axes[0, 0].set_xlabel("fitted"); axes[0, 0].set_ylabel("residual")
stats.probplot(std_resid, dist="norm", plot=axes[0, 1]); axes[0, 1].set_title("Normal Q-Q")
axes[1, 0].scatter(fitted, np.sqrt(np.abs(std_resid)), alpha=0.3, s=8)
axes[1, 0].set_title("Scale-location"); axes[1, 0].set_xlabel("fitted")
sns.histplot(resid, kde=True, ax=axes[1, 1]); axes[1, 1].set_title("Residual histogram")
plt.tight_layout()
plt.show()
# Formal tests
print("Shapiro-Wilk (normality of residuals) p =", round(stats.shapiro(resid[:5000])[1], 4))
dw = np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2)
print("Durbin-Watson (~2 = no autocorrelation):", round(dw, 3))
Xc = np.column_stack([np.ones(len(X_train)), X_train.values])
aux = LinearRegression().fit(Xc, resid ** 2) # Breusch-Pagan test
lm = len(resid) * aux.score(Xc, resid ** 2)
print("Breusch-Pagan p (heteroscedasticity):", round(1 - stats.chi2.cdf(lm, p - 1), 4))
# ---------- 5. Cook's distance (influential observations) ----------
cooks = (resid ** 2 / (p * mse)) * (h / (1 - h) ** 2)
thresh = 4 / len(cooks)
influential = np.where(cooks > thresh)[0]
print(f"Cook's D threshold (4/n) = {thresh:.5f} | influential points: {len(influential)}")
print("Top 10 by Cook's D:\n", pd.Series(cooks, index=X_train.index).nlargest(10).round(5))
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].stem(cooks, markerfmt=",", basefmt=" ")
axes[0].axhline(thresh, color="r", ls="--", label="4/n"); axes[0].legend()
axes[0].set_title("Cook's distance"); axes[0].set_xlabel("observation")
axes[1].scatter(h, std_resid, alpha=0.3, s=8)
axes[1].set_xlabel("leverage"); axes[1].set_ylabel("studentised residual")
axes[1].set_title("Residuals vs leverage")
plt.tight_layout()
plt.show()
# ---------- 6. Refit without influential points ----------
keep = np.setdiff1d(np.arange(len(X_train)), influential)
table2 = ols_fit(X_train.iloc[keep], y_train.iloc[keep])[0]
print(pd.concat([table["coef"], table2["coef"]], axis=1,
 keys=["all data", "without influential"]).round(4))
# Optional: the same output using statsmodels (if installed)
# import statsmodels.api as sm
# model = sm.OLS(y_train, sm.add_constant(X_train)).fit(); print(model.summary())
# cooks_sm = model.get_influence().cooks_distance[0]