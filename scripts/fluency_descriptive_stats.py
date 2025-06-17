import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Resolve correct paths
project_root = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(project_root, "data")
plots_dir = os.path.join(project_root, "plots")
os.makedirs(plots_dir, exist_ok=True)

input_file = os.path.join(data_dir, "Averaged_Evaluations.csv")
plot_file = os.path.join(plots_dir, "fluency_boxplot_by_model_age.png")

# Load the averaged evaluation data
df = pd.read_csv(input_file)

# Ensure Age is treated as categorical (for consistent plotting order)
df["Age"] = df["Age"].astype(str)

# Compute and print mean and std for each Model x Age group
summary_stats = df.groupby(["Model", "Age"])["Fluency"].agg(["mean", "std"]).reset_index()
print("Fluency Score Summary (Mean and Std by Model and Age Group):\n")
print(summary_stats.to_string(index=False))

# Set plot style
sns.set(style="whitegrid")

# Create boxplots of Fluency by Model and Age
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="Age", y="Fluency", hue="Model")

plt.title("Fluency Scores by Model and Age Group")
plt.ylabel("Fluency (Avg of 4 Metrics)")
plt.xlabel("Age Group")
plt.ylim(0.5, 5.5)
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

# Save plot
plt.savefig(plot_file)
