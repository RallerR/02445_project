import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Set paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "..", "data")
data_path = os.path.join(data_dir, "Averaged_Evaluations.csv")

# Load data
df = pd.read_csv(data_path)

# Create output folder for plots (optional)
plot_dir = os.path.join(base_dir, "..", "plots", "qqplots")
os.makedirs(plot_dir, exist_ok=True)

# Create Q-Q plots for each Model × Age group
grouped = df.groupby(["Model", "Age"])

for (model, age), group in grouped:
    plt.figure(figsize=(6, 6))
    stats.probplot(group["Fluency"], dist="norm", plot=plt)
    plt.title(f"Q-Q Plot: {model}, Age {age}")
    plt.grid(True)
    plt.tight_layout()

    # Optional: save each plot
    safe_model = model.replace("/", "_").replace(" ", "_")
    filename = f"qqplot_{safe_model}_age{age}.png"
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()
