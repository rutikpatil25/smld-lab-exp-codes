import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('smart_campus_dataset_10K.csv')

# Outlier detection using IQR
Q1 = df['capacity'].quantile(0.25)
Q3 = df['capacity'].quantile(0.75)
IQR = Q3 - Q1

lower_boundary = Q1 - 1.5 * IQR
upper_boundary = Q3 + 1.5 * IQR

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower boundary:", lower_boundary)
print("Upper boundary:", upper_boundary)

# Display outliers
outliers = df[(df['capacity'] < lower_boundary) |
              (df['capacity'] > upper_boundary)]

print("Outliers:")
print(outliers)

# Minimum and maximum capacity
print("Minimum Capacity:", df['capacity'].min())
print("Maximum Capacity:", df['capacity'].max())

# Box plot
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='capacity', color='skyblue')
plt.title('Box Plot of Capacity (Detecting Outliers)')
plt.show()