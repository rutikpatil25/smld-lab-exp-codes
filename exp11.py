import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet,
 RidgeCV, LassoCV, ElasticNetCV)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
df = pd.read_csv("smart_campus_dataset_10K.csv")
df["net_flow"] = df["entry_count"] - df["exit_count"] # collinear on purpose
df["total_flow"] = df["entry_count"] + df["exit_count"]
target = "occupancy_count"
features = ["capacity", "idle_minutes", "power_kwh", "hvac_state", "indoor_temperature",
 "outdoor_temperature_index", "cpu_utilization", "power_draw", "runtime_hours",
 "device_temperature", "connected_devices", "bandwidth_usage", "density_index",
 "entry_count", "exit_count", "net_flow", "total_flow"]
# ---------- 1. Split and standardise (penalties need features on one scale) ----------
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target],
 test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
# ---------- 2. Choose hyper-parameters by 5-fold cross-validation ----------
ridge_cv = RidgeCV(alphas=np.logspace(-3, 4, 50), cv=5).fit(Xtr, y_train)
lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 50), cv=5, random_state=42,
 max_iter=20000).fit(Xtr, y_train)
enet_cv = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95], alphas=np.logspace(-4, 1, 50),
 cv=5, random_state=42, max_iter=20000).fit(Xtr, y_train)
print("Best Ridge alpha :", round(ridge_cv.alpha_, 4))
print("Best Lasso alpha :", round(lasso_cv.alpha_, 4))
print("Best ElasticNet alpha :", round(enet_cv.alpha_, 4), "| l1_ratio:", enet_cv.l1_ratio_)
# ---------- 3. Fit final models ----------
models = {
 "OLS": LinearRegression().fit(Xtr, y_train),
 "Ridge": Ridge(alpha=ridge_cv.alpha_).fit(Xtr, y_train),
 "Lasso": Lasso(alpha=lasso_cv.alpha_, max_iter=20000).fit(Xtr, y_train),
 "ElasticNet": ElasticNet(alpha=enet_cv.alpha_, l1_ratio=enet_cv.l1_ratio_,
 max_iter=20000).fit(Xtr, y_train),
}
# ---------- 4. Performance comparison ----------
rows = []
for name, m in models.items():
 rows.append([name,
 r2_score(y_train, m.predict(Xtr)), r2_score(y_test, m.predict(Xte)),
 np.sqrt(mean_squared_error(y_test, m.predict(Xte))),
 int(np.sum(np.abs(m.coef_) > 1e-8))])
perf = pd.DataFrame(rows, columns=["model", "train R2", "test R2", "test RMSE",
 "non-zero coefs"])
print(perf.round(4))
# ---------- 5. Coefficient comparison ----------
coefs = pd.DataFrame({n: m.coef_ for n, m in models.items()}, index=features)
print(coefs.round(3))
print("Features removed by Lasso :", list(coefs.index[coefs["Lasso"].abs() < 1e-8]))
print("Features removed by ElasticNet:", list(coefs.index[coefs["ElasticNet"].abs() < 1e-8]))
coefs.plot(kind="bar", figsize=(15, 6), width=0.8)
plt.axhline(0, color="k", lw=0.8); plt.ylabel("standardised coefficient")
plt.title("OLS vs Ridge vs Lasso vs Elastic Net")
plt.tight_layout()
plt.show()
# ---------- 6. Regularisation paths ----------
alphas = np.logspace(-3, 3, 60)
ridge_path = np.array([Ridge(alpha=a).fit(Xtr, y_train).coef_ for a in alphas])
lasso_path = np.array([Lasso(alpha=a / 10, max_iter=20000).fit(Xtr, y_train).coef_
 for a in alphas])
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, path, title in [(axes[0], ridge_path, "Ridge path"),
 (axes[1], lasso_path, "Lasso path")]:
 ax.plot(alphas, path)
 ax.set_xscale("log"); ax.set_xlabel("alpha"); ax.set_ylabel("coefficient")
 ax.set_title(title)
plt.tight_layout()
plt.show()
# ---------- 7. Cross-validation error curve for Lasso ----------
plt.figure(figsize=(8, 5))
plt.semilogx(lasso_cv.alphas_, lasso_cv.mse_path_.mean(axis=1), "b-")
plt.axvline(lasso_cv.alpha_, color="r", ls="--", label="best alpha")
plt.xlabel("alpha"); plt.ylabel("CV mean squared error"); plt.legend()
plt.title("LassoCV error curve")
plt.show()