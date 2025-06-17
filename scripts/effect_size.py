import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path
import os

# Define paths relative to this script's directory
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
data_dir = project_root / "data"
os.makedirs(data_dir, exist_ok=True)

# Load the dataset
df = pd.read_csv(data_dir / "Averaged_Evaluations.csv")

# Fit the two-way ANOVA model with interaction
model = smf.ols("Fluency ~ C(Model) * C(Age)", data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# Display ANOVA results
print("=== Two-Way ANOVA ===")
print(anova_table)

# ---- Compute Eta Squared (η²) and Partial Eta Squared ----
ss_total = anova_table["sum_sq"].sum()
ss_residual = anova_table.loc["Residual", "sum_sq"]

anova_table["eta_sq"] = anova_table["sum_sq"] / ss_total
anova_table["partial_eta_sq"] = anova_table["sum_sq"] / (anova_table["sum_sq"] + ss_residual)

# Format effect sizes
effect_sizes = anova_table[["sum_sq", "eta_sq", "partial_eta_sq"]].round(4)

# Print effect sizes
print("\n=== Effect Sizes ===")
print(effect_sizes)

# Save to CSV
effect_sizes.to_csv(data_dir / "TwoWayANOVA_EffectSizes.csv", index=True)
