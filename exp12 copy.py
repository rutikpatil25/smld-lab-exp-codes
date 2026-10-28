
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("smart_campus_dataset_10K.csv")
df["high_util"] = (df["utilization_ratio"] >= 0.5).astype(int)

# Features and target
X = df[["capacity", "occupancy_count", "idle_minutes",
        "power_kwh", "connected_devices"]]
y = df["high_util"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features for Logistic Regression and KNN
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

# Create models
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB()
}

# Train and evaluate
for name, model in models.items():
    if name == "Naive Bayes":
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
    else:
        model.fit(X_train_sc, y_train)
        pred = model.predict(X_test_sc)

    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
