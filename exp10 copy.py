import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# Sample Data
data = pd.DataFrame({
    'X1': [1, 2, 3, 4, 5],
    'X2': [2, 3, 4, 5, 6],
    'Y': [3, 5, 7, 9, 11]
})

# Linear Regression
X = data[['X1', 'X2']]
Y = data['Y']

model = LinearRegression()
model.fit(X, Y)
pred = model.predict(X)

# VIF
X_vif = sm.add_constant(X)
vif = pd.DataFrame()
vif["Variable"] = X_vif.columns
vif["VIF"] = [
    variance_inflation_factor(X_vif.values, i)
    for i in range(X_vif.shape[1])
]
print("VIF")
print(vif)

# Residual Plot
residual = Y - pred
plt.scatter(pred, residual)
plt.axhline(y=0, color='red')
plt.title("Residual Plot")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.show()

# Cook's Distance
model2 = sm.OLS(Y, X_vif).fit()
influence = model2.get_influence()
cook = influence.cooks_distance[0]

print("Cook's Distance")
print(cook)