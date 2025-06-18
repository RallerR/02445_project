import pandas as pd
import os
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro

# Resolve project paths
project_root = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(project_root, "data")
plot_dir = os.path.join(project_root, "plots")
os.makedirs(plot_dir, exist_ok=True)

input_file = os.path.join(data_dir, "Averaged_Evaluations.csv")
output_plot = os.path.join(plot_dir, "anova_residuals_distribution.png")

# Load data
df = pd.read_csv(input_file)

# Fit the two-way ANOVA model
model = smf.ols("Fluency ~ C(Model) * C(Age)", data=df).fit()

# Get residuals
residuals = model.resid

# Plot and save histogram of residuals
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, bins=20, color='skyblue')
plt.title("Distribution of ANOVA Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(output_plot)
plt.close()
print(f"Saved plot to: {output_plot}")

# Perform Shapiro-Wilk test
stat, p_value = shapiro(residuals)
print(f"Shapiro-Wilk test: W={stat:.4f}, p-value={p_value:.4f}")
if p_value < 0.05:
    print("Residuals are likely not normally distributed.")
else:
    print("Residuals appear to be normally distributed.")
