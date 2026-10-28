import pandas as pd

# Load the dataset
df = pd.read_csv('smart_campus_dataset_10K.csv')

# 1. Simple Random Sampling (SRS)
srs = df.sample(n=100, random_state=42)
print("Simple Random Sample:")
print(srs.head())

# 2. Stratified Sampling
# We use 'space_id' as our target column to group the data
stratified = df.groupby('space_id', group_keys=False).apply(
    lambda x: x.sample(frac=0.2, random_state=42)
)
print("\nStratified Sample:")
print(stratified.head())

# 3. Cluster Sampling
# We pick the first 2 unique space IDs as our clusters
clusters = df['space_id'].unique()[:2]
cluster = df[df['space_id'].isin(clusters)]
print("\nCluster Sample:")
print(cluster.head())
