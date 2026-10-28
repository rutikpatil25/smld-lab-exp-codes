from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# Create sample data
X, y = make_regression(
    n_samples=100,
    n_features=2,
    noise=10,
    random_state=42
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print("Ridge Score:", round(ridge.score(X_test, y_test), 2))

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print("Lasso Score:", round(lasso.score(X_test, y_test), 2))

# Elastic Net Regression
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train, y_train)
print("Elastic Net Score:", round(elastic.score(X_test, y_test), 2))