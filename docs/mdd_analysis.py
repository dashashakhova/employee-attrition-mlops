import numpy as np
from scipy.stats import ttest_ind

np.random.seed(42)

existing_system_responses = np.random.normal(loc=3.5, scale=0.4, size=500_000)
improved_system_responses = np.random.normal(loc=2.0, scale=0.4, size=500_000)

result = ttest_ind(
    existing_system_responses,
    improved_system_responses,
    equal_var=False,
    alternative="greater",
)

print(f"t-statistic: {result.statistic:.4f}")
print(f"p-value: {result.pvalue}")
print("Decision: reject H0" if result.pvalue < 0.05 else "Decision: do not reject H0")