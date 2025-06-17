from statsmodels.stats.power import FTestAnovaPower
import numpy as np

# Mean scores per model
means = [4.55, 4.56, 4.63]

# Pooled standard deviation (simple average since groups are assumed balanced)
stds = [0.18, 0.16, 0.13]
pooled_std = np.mean(stds)

grand_mean = np.mean(means)
ss_between = sum((m - grand_mean) ** 2 for m in means) / len(means)

# Calculate the effect size (Cohen's f)
effect_size_f = np.sqrt(ss_between) / pooled_std

print(f"Estimated Cohen's f: {effect_size_f:.4f}")

# Estimate required sample size per group
analysis = FTestAnovaPower()
sample_size = analysis.solve_power(
    effect_size=effect_size_f,
    alpha=0.05,
    power=0.8,
    k_groups=3
)

print(f"Required sample size per group: {np.ceil(sample_size)}")
