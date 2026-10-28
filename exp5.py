import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("smart_campus_dataset_10K.csv")

print("Dataset Head:")
print(df.head())

# Extracting hours column
hours = df['hour_index']
print("\nHour Index Head:")
print(hours.head())

# 2. BEFORE CLT
df_population = pd.DataFrame({'Hour_Index': hours})
pop_mean = df_population['Hour_Index'].mean()

# 3. AFTER CLT
np.random.seed(42)
num_samples = 2000
sample_size = 50
sample_means = []

for _ in range(num_samples):
    sample = df_population['Hour_Index'].sample(
        n=sample_size, replace=True
    )
    sample_means.append(sample.mean())

df_clt = pd.DataFrame({'Sample_Means': sample_means})

# Calculating sample mean
clt_mean = df_clt['Sample_Means'].mean()

# 4. Plotting
plt.hist(df_population['Hour_Index'], bins=30,
         color='crimson', edgecolor='black', alpha=0.7)
plt.title('Before CLT: Original Data\n(on hour_index column)')
plt.axvline(pop_mean, color='black', linestyle='dashed',
            linewidth=2, label=f'Mean: {pop_mean:.2f}')
plt.text(pop_mean, plt.gca().get_ylim()[1] * 0.9,
         f'Mean: {pop_mean:.2f}', color='black', fontweight='bold')
plt.xlabel('Hour Index')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# After CLT
plt.hist(df_clt['Sample_Means'], bins=30,
         color='royalblue', edgecolor='black', alpha=0.7)
plt.title('After CLT: Distribution of Sample Means')
plt.axvline(clt_mean, color='black', linestyle='dashed',
            linewidth=2, label=f'Mean: {clt_mean:.2f}')
plt.text(clt_mean, plt.gca().get_ylim()[1] * 0.9,
         f'Mean: {clt_mean:.2f}', color='black', fontweight='bold')
plt.xlabel('Sample Mean of Hour Index')
plt.ylabel('Frequency')
plt.legend()
plt.show()