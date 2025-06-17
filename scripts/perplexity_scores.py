import os
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- Set up paths ---
project_root = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(project_root, "data")
os.makedirs(data_dir, exist_ok=True)

input_file = os.path.join(data_dir, "Averaged_Evaluations_with_Readability.csv")
output_file = os.path.join(data_dir, "Averaged_Evaluations_with_Perplexity.csv")
summary_file = os.path.join(data_dir, "Perplexity_Summary_by_Model_and_Age.csv")
pivot_file = os.path.join(data_dir, "Perplexity_Comparison_Table.csv")

# --- Load data ---
df = pd.read_csv(input_file)

# --- Load GPT-2 model ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
if torch.cuda.is_available():
    model = model.cuda()

# --- Function to compute perplexity ---
def compute_perplexity(text):
    try:
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encodings.input_ids
        with torch.no_grad():
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        return torch.exp(loss).item()
    except Exception as e:
        print(f"Error computing perplexity: {e}")
        return None

# --- Compute perplexity ---
tqdm.pandas(desc="Computing GPT-2 Perplexity")
df["GPT2_Perplexity"] = df["Response"].progress_apply(compute_perplexity)

# --- Save updated dataset ---
df.to_csv(output_file, index=False)

# --- Summary statistics ---
summary = df.groupby(["Model", "Age"])["GPT2_Perplexity"].agg(["mean", "std"]).round(2).reset_index()
pivot = summary.pivot(index="Age", columns="Model", values="mean").round(2)

# --- Print summaries ---
print("\n Average GPT-2 Perplexity per Model and Age:")
print(summary)

print("\n Pivot Table (Age × Model Perplexity Mean):")
print(pivot)

# --- Save summaries ---
summary.to_csv(summary_file, index=False)
pivot.to_csv(pivot_file)




# --- Plot directory setup ---
plots_dir = os.path.join(project_root, "plots")
os.makedirs(plots_dir, exist_ok=True)
plot_file = os.path.join(plots_dir, "perplexity_by_model_and_age.png")

# --- Plot mean perplexity ---
plt.figure(figsize=(10, 6))
sns.barplot(data=summary, x="Age", y="mean", hue="Model")

plt.title("Mean GPT-2 Perplexity by Model and Age Group")
plt.ylabel("Mean Perplexity (GPT-2)")
plt.xlabel("Age Group")
plt.ylim(0, summary["mean"].max() + 5)
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

plt.savefig(plot_file)
plt.close()
