
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    ConfusionMatrixDisplay
)


# Load dataset
df = pd.read_csv("smart_campus_dataset_10K.csv")


# 1. Create binary target
df["high_util"] = (
    df["utilization_ratio"] >= 0.5
).astype(int)

print(df["high_util"].value_counts(normalize=True).round(3))


# Select features
features = [
    "hour_index",
    "day_index",
    "capacity",
    "occupancy_count",
    "idle_minutes",
    "power_kwh",
    "hvac_state",
    "indoor_temperature",
    "outdoor_temperature_index",
    "cpu_utilization",
    "power_draw",
    "runtime_hours",
    "connected_devices",
    "bandwidth_usage",
    "entry_count",
    "exit_count",
    "density_index"
]


# Split dataset
# utilization_ratio is excluded to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    df[features],
    df["high_util"],
    test_size=0.2,
    random_state=42,
    stratify=df["high_util"]
)


# 2. Find the best K for KNN using 5-fold CV
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

ks = range(1, 31, 2)

cv_acc = [
    cross_val_score(
        make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(k)
        ),
        X_train,
        y_train,
        cv=cv
    ).mean()
    for k in ks
]

best_k = list(ks)[int(np.argmax(cv_acc))]

print(
    "Best k =", best_k,
    "| CV accuracy =", round(max(cv_acc), 4)
)


# Plot KNN accuracy
plt.figure(figsize=(8, 4.5))
plt.plot(list(ks), cv_acc, marker="o")
plt.axvline(best_k, color="r", linestyle="--")
plt.xlabel("k")
plt.ylabel("CV accuracy")
plt.title("KNN: Choosing k")
plt.show()


# 3. Define models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    ),
    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(best_k)
    ),
    "Naive Bayes": GaussianNB()
}


# 4. Train and evaluate models
results = []
probs = {}

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

for ax, (name, model) in zip(axes, models.items()):

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    probs[name] = prob

    results.append([
        name,
        accuracy_score(y_test, pred),
        precision_score(y_test, pred),
        recall_score(y_test, pred),
        f1_score(y_test, pred),
        roc_auc_score(y_test, prob),
        cross_val_score(
            model, X_train, y_train, cv=cv
        ).mean()
    ])

    print(f"\n===== {name} =====")

    print(confusion_matrix(y_test, pred))

    print(
        classification_report(
            y_test,
            pred,
            target_names=["low util", "high util"]
        )
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        pred,
        ax=ax,
        colorbar=False
    )

    ax.set_title(name)

plt.tight_layout()
plt.show()


# Display comparison of models
summary = pd.DataFrame(
    results,
    columns=[
        "model",
        "accuracy",
        "precision",
        "recall",
        "F1",
        "ROC-AUC",
        "CV accuracy"
    ]
)

print(summary.round(4))


# 5. Plot ROC curves
plt.figure(figsize=(7, 6))

for name, prob in probs.items():

    fpr, tpr, _ = roc_curve(y_test, prob)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {roc_auc_score(y_test, prob):.3f})"
    )

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()


# 6. Logistic Regression coefficients and odds ratios
logit = models["Logistic Regression"]

coef = pd.DataFrame(
    {
        "coef": logit[-1].coef_[0],
        "odds_ratio": np.exp(logit[-1].coef_[0])
    },
    index=features
)

print(
    coef.sort_values(
        "coef",
        key=abs,
        ascending=False
    ).to_string(float_format="{:.4g}".format)
)


# 7. Naive Bayes class priors and feature means
nb = models["Naive Bayes"]

print("Class priors:", nb.class_prior_.round(3))

print(
    pd.DataFrame(
        nb.theta_,
        columns=features,
        index=["low util", "high util"]
    ).T.round(3)
)
