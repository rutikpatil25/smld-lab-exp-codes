import math
import matplotlib.pyplot as plt

# Descriptive Statistics (Mean & Standard Deviation)
data = [16, 19, 28, 37, 61]

# Mean
mean = sum(data) / len(data)

# Variance
variance = sum((x - mean) ** 2 for x in data) / len(data)

# Standard Deviation
std = math.sqrt(variance)

print("Mean =", mean)
print("Standard Deviation =", round(std, 2))

# MLE for Bernoulli Distribution
data_bernoulli = [1, 0, 1, 1, 0, 1, 1]

p = sum(data_bernoulli) / len(data_bernoulli)

print("Estimated Probability (p) =", round(p, 2))

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Descriptive Stats Histogram (Using index)
axes[0].hist(data, bins=5, color='salmon',
             edgecolor='black', alpha=0.7)
axes[0].axvline(mean, color='blue', linestyle='--',
                linewidth=2, label=f'Mean ({mean:.1f})')
axes[0].axvline(mean - std, color='purple', linestyle=':',
                linewidth=2, label=f'-1 SD ({mean-std:.1f})')
axes[0].axvline(mean + std, color='purple', linestyle=':',
                linewidth=2, label=f'+1 SD ({mean+std:.1f})')
axes[0].set_title("Histogram of Original Data")
axes[0].set_xlabel("Values")
axes[0].set_ylabel("Frequency")
axes[0].legend()

# Bernoulli Distribution Chart (Using index)
zeros = data_bernoulli.count(0)
ones = data_bernoulli.count(1)

axes[1].bar(['0 (Failure)', '1 (Success)'], [zeros, ones],
            color=['gold', 'lightseagreen'],
            edgecolor='black', width=0.4)
axes[1].set_title(f"Bernoulli Trials (p = {p:.2f})")
axes[1].set_xlabel("Outcomes")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()