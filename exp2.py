import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

df = pd.read_csv('smart_campus_dataset_10K.csv')

print(df.head())

# step 1
# missing values
print("Missing values:")
print(df.isnull().sum())

# in percent for capacity
print("Missing value percentage:")
print((df['capacity'].isnull().sum() / len(df)) * 100)

# step 2
# replace null values with mean
mean = df['capacity'].mean()
df['capacity'] = df['capacity'].fillna(mean)

# check again
print("Missing values in capacity:", df['capacity'].isnull().sum())
print(df['capacity'])

# step 3
# scaling
# handle missing values before scaling
df['occupancy_count'] = df['occupancy_count'].fillna(
    df['occupancy_count'].median()
)

columns_to_scale = ['capacity', 'occupancy_count']

min_max_scaler = MinMaxScaler()
df_normalized = df.copy()

df_normalized[columns_to_scale] = min_max_scaler.fit_transform(
    df[columns_to_scale]
)

print("Normalized Data (0 to 1)")
print(df_normalized[columns_to_scale].head())

# step 4
# encoding
categorical_columns = df_normalized.select_dtypes(
    include=['object', 'category']
).columns

for column in categorical_columns:
    le = LabelEncoder()
    df_normalized[column] = le.fit_transform(
        df_normalized[column].astype(str)
    )

print("Encoded Data:")
print(df_normalized[categorical_columns].head())

# step 5
# outlier detection using IQR on original capacity values
Q1 = df['capacity'].quantile(0.25)
Q3 = df['capacity'].quantile(0.75)
IQR = Q3 - Q1

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)

# calculate boundaries
lower_boundary = Q1 - 1.5 * IQR
upper_boundary = Q3 + 1.5 * IQR

print("Lower boundary:", lower_boundary)
print("Upper boundary:", upper_boundary)

# display outliers
outliers = df[
    (df['capacity'] < lower_boundary) |
    (df['capacity'] > upper_boundary)
]

print("Outliers:")
print(outliers)

# minimum and maximum capacity
print("Minimum Capacity:", df['capacity'].min())
print("Maximum Capacity:", df['capacity'].max())

# box plot
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='capacity', color='skyblue')
plt.title('Box Plot of Capacity (Detecting Outliers)')
plt.xlabel('Capacity')
plt.show()