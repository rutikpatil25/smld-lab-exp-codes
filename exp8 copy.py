import random
import matplotlib.pyplot as plt

print("Initializing Experiment 6\n")

# 1. Bootstrap
data = [16, 19, 28, 37, 61]
means = []

# Bootstrap sampling
for i in range(1000):
    sample = random.choices(data, k=len(data))
    means.append(sum(sample) / len(sample))

# Sort and print 95% confidence interval
means.sort()
lower_ci = means[25]
upper_ci = means[975]

print("Bootstrap (95% Confidence Interval):")
print(lower_ci, "to", upper_ci)
print()

# 2. Permutation Test
# Two groups
group1 = [24, 26, 28]
group2 = [34, 36, 38]

# Original difference in means
original = abs(sum(group1) / len(group1) -
               sum(group2) / len(group2))

# Combine both groups
combined = group1 + group2
all_diffs = []

# Perform permutation test
count = 0

for i in range(1000):
    random.shuffle(combined)
    new1 = combined[:3]
    new2 = combined[3:]

    diff = abs(sum(new1) / len(new1) -
               sum(new2) / len(new2))

    all_diffs.append(diff)

    if diff >= original:
        count += 1

p_value = count / 1000

print("Permutation Test:")
print("Original Difference =", original)
print("P-value =", p_value)

# 3. Histograms
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bootstrap Sample Means Histogram
axes[0].hist(means, bins=20, color='skyblue', edgecolor='black')
axes[0].set_title("Bootstrap Distribution of Means")
axes[0].set_xlabel("Sample Mean")
axes[0].set_ylabel("Frequency")

# Permutation Test Differences Histogram
axes[1].hist(all_diffs, bins=10, color='lightgreen', edgecolor='black')
axes[1].axvline(original, color='red', linestyle='--',
                linewidth=2, label=f'Original Diff: {original:.2f}')
axes[1].set_title("Permutation Test Differences")
axes[1].set_xlabel("Absolute Difference in Means")
axes[1].set_ylabel("Frequency")
axes[1].legend()

plt.tight_layout()
plt.show()