import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set base and data directories
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "..", "data")
plot_dir = os.path.join(base_dir, "..", "plots")
os.makedirs(plot_dir, exist_ok=True)

# Load the averaged evaluation file
data_path = os.path.join(data_dir, "Averaged_Evaluations.csv")
df = pd.read_csv(data_path)

# --- Descriptive statistics ---

# By model
desc_model = df.groupby("Model")["Fluency"].agg(["mean", "std"]).reset_index()
desc_model = desc_model.round(2)
desc_model.to_csv(os.path.join(data_dir, "fluency_by_model.csv"), index=False)

# By age group
desc_age = df.groupby("Age")["Fluency"].agg(["mean", "std"]).reset_index()
desc_age = desc_age.round(2)
desc_age.to_csv(os.path.join(data_dir, "fluency_by_age.csv"), index=False)

# By model × age group
desc_model_age = df.groupby(["Model", "Age"])["Fluency"].agg(["mean", "std"]).reset_index()
desc_model_age = desc_model_age.round(2)
desc_model_age.to_csv(os.path.join(data_dir, "fluency_by_model_age.csv"), index=False)

# --- Prepare renamed model names for plotting ---

model_name_map = {
    "claude-3-7-sonnet-20250219": "Claude 3 Sonnet",
    "gpt-4o": "GPT-4o",
    "models/gemini-2.0-flash": "Gemini 2 Flash"
}

# Copy dataframe for plotting with clean model names
df_plot = df.copy()
df_plot["Model"] = df_plot["Model"].map(model_name_map)

# --- Visualizations ---

sns.set(style="whitegrid")

# Fluency by Model
plt.figure(figsize=(10, 6))
sns.boxplot(x="Model", y="Fluency", data=df_plot)
plt.xticks(rotation=30, ha="right")
plt.title("Fluency by Model")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "boxplot_fluency_by_model.png"))
plt.close()

# Fluency by Age
plt.figure(figsize=(8, 6))
sns.boxplot(x="Age", y="Fluency", data=df_plot)
plt.title("Fluency by Age Group")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "boxplot_fluency_by_age.png"))
plt.close()

# Fluency by Model × Age Group
plt.figure(figsize=(12, 6))
sns.boxplot(x="Age", y="Fluency", hue="Model", data=df_plot)
plt.title("Fluency by Age Group and Model")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "boxplot_fluency_by_model_age.png"))
plt.close()
