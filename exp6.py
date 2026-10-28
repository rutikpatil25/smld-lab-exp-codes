import pandas as pd
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway

# Load data
df = pd.read_csv('smart_campus_dataset_10K.csv')

# T-test
# Compare power consumption for HVAC ON and OFF
on = df[df['hvac_state'] == 1]['power_kwh']
off = df[df['hvac_state'] == 0]['power_kwh']

t, p = ttest_ind(on, off)

print("T-statistic:", t)
print("P-value:", p)

if p < 0.05:
    print("Reject Null Hypothesis")
else:
    print("Accept Null Hypothesis")

# Chi-square test
# Convert utilization into categories
df['utilization'] = pd.cut(
    df['utilization_ratio'],
    bins=[-0.01, 0.5, 1.0],
    labels=['Low', 'High']
)

table = pd.crosstab(df['hvac_state'], df['utilization'])

chi, p, dof, expected = chi2_contingency(table)

print("Chi-square:", chi)
print("P-value:", p)

if p < 0.05:
    print("Reject Null Hypothesis")
else:
    print("Accept Null Hypothesis")

# Anova
# Divide utilization into three groups
df['utilization'] = pd.cut(
    df['utilization_ratio'],
    bins=[-0.01, 0.33, 0.66, 1.0],
    labels=['Low', 'Medium', 'High']
)

low = df[df['utilization'] == 'Low']['power_kwh']
medium = df[df['utilization'] == 'Medium']['power_kwh']
high = df[df['utilization'] == 'High']['power_kwh']

f, p = f_oneway(low, medium, high)

print("F-statistic:", f)
print("P-value:", p)

if p < 0.05:
    print("Reject Null Hypothesis")
else:
    print("Accept Null Hypothesis")