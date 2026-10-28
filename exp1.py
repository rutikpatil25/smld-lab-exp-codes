import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('smart_campus_dataset_10K.csv')

# taking capacity column for experimentation
data = df['capacity']

# mean
mean = data.mean()
print(f"mean :{mean:.2f}")

# median
median = data.median()
print(f"median :{median:.2f}")

# mode
mode = data.mode()
if mode.empty:
    print("mode : No Data Available")
else:
    mod = mode[0]  # Safely extract the first mode value
    print(f"mode :{mod:.2f}")

plt.hist(data, bins=15, edgecolor='black')
plt.axvline(mean, color='skyblue', linestyle='--', label=f'Mean: {mean:.2f}')
plt.axvline(median, color='green', linestyle='-', label=f'Median: {median:.2f}')

if mode.empty:
    print("mode : No Data Available")
else:
    mod = mode[0]  # Safely extracted scalar value
    plt.axvline(mod, color='red', linestyle='--', label=f'Mode: {mod:.2f}')

plt.legend()
plt.title("Histogram of Capacity column")
plt.xlabel("capacity")
plt.ylabel("count")
plt.show()

# correlation matrix betn capacity,occupancy_count
capacity = df['capacity']
occ_cnt = df['occupancy_count']
matrix = pd.DataFrame({'capacity': capacity, 'occupancy_count': occ_cnt}).corr()
print("Corelation matrix")
print(matrix)