import pandas as pd
from scipy.stats import kruskal
import os

# Set project paths
project_root = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(project_root, "data")
input_file = os.path.join(data_dir, "Averaged_Evaluations.csv")

# Load data
df = pd.read_csv(input_file)

# === Kruskal-Wallis: Comparing Models at Each Age ===
results_by_age = []

for age in sorted(df["Age"].unique()):
    subset = df[df["Age"] == age]
    models = sorted(subset["Model"].unique())
    groups = [subset[subset["Model"] == model]["Fluency"] for model in models]
    stat, p = kruskal(*groups)
    results_by_age.append({"Age": age, "H-statistic": round(stat, 4), "p-value": round(p, 4)})

# Save table
df_results_by_age = pd.DataFrame(results_by_age)
df_results_by_age.to_csv(os.path.join(data_dir, "Kruskal_by_Age.csv"), index=False)

# === Kruskal-Wallis: Comparing Age Groups for Each Model ===
results_by_model = []

for model in sorted(df["Model"].unique()):
    subset = df[df["Model"] == model]
    ages = sorted(subset["Age"].unique())
    groups = [subset[subset["Age"] == age]["Fluency"] for age in ages]
    stat, p = kruskal(*groups)
    results_by_model.append({"Model": model, "H-statistic": round(stat, 4), "p-value": round(p, 4)})

# Save table
df_results_by_model = pd.DataFrame(results_by_model)
df_results_by_model.to_csv(os.path.join(data_dir, "Kruskal_by_Model.csv"), index=False)

# Optional: Print results
print("=== Kruskal-Wallis: Comparing Models at Each Age ===")
print(df_results_by_age, "\n")

print("=== Kruskal-Wallis: Comparing Age Groups for Each Model ===")
print(df_results_by_model)
