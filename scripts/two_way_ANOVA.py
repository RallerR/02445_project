import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# --- Setup paths relative to the script's location ---
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
data_dir = project_root / "data"
plots_dir = project_root / "plots"
os.makedirs(plots_dir, exist_ok=True)

# Load data
df = pd.read_csv(data_dir / "Averaged_Evaluations.csv")

# Two-way ANOVA
df["Model"] = df["Model"].astype("category")
df["Age"] = df["Age"].astype("category")

model = ols("Fluency ~ C(Model) + C(Age) + C(Model):C(Age)", data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# Print ANOVA results
print("\nTwo-Way ANOVA Results:")
print(anova_table)

# Save to CSV
anova_table.to_csv(data_dir / "Two_Way_ANOVA_Results.csv")

# Plotting
plt.figure(figsize=(8, 6))
sns.pointplot(data=df, x="Age", y="Fluency", hue="Model", ci="sd", dodge=True, markers=["o", "s", "D"])
plt.title("Fluency Scores by Model and Age Group")
plt.xlabel("Age Group")
plt.ylabel("Fluency (Average of 4 Metrics)")
plt.ylim(0.5, 5.5)
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

# Save plot
plot_file = plots_dir / "fluency_two_way_anova_plot.png"
plt.savefig(plot_file)
plt.close()

