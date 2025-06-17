import os
import pandas as pd
import textstat

# Resolve correct paths
project_root = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(project_root, "data")
os.makedirs(data_dir, exist_ok=True)

input_file = os.path.join(data_dir, "Averaged_Evaluations.csv")
output_file = os.path.join(data_dir, "Averaged_Evaluations_with_Readability.csv")
table_fre_file = os.path.join(data_dir, "Flesch_Reading_Ease_Table.csv")
table_fkg_file = os.path.join(data_dir, "Flesch_Kincaid_Grade_Table.csv")

# Load the averaged evaluations
df = pd.read_csv(input_file)

# Compute readability scores
df["Flesch_Reading_Ease"] = df["Response"].apply(textstat.flesch_reading_ease)
df["Flesch_Kincaid_Grade"] = df["Response"].apply(textstat.flesch_kincaid_grade)

# Save enriched data
df.to_csv(output_file, index=False)
print(f"Saved enriched data with readability scores to: {output_file}")

# Group by Model and Age and compute mean
grouped = df.groupby(["Age", "Model"])[["Flesch_Reading_Ease", "Flesch_Kincaid_Grade"]].mean().round(2).reset_index()

# Pivot to match desired table structure
table_fre = grouped.pivot(index="Age", columns="Model", values="Flesch_Reading_Ease")
table_fkg = grouped.pivot(index="Age", columns="Model", values="Flesch_Kincaid_Grade")

# Print summary
print("\n Flesch Reading Ease (higher = easier):")
print(table_fre)

print("\n Flesch-Kincaid Grade Level (higher = more difficult):")
print(table_fkg)

# Save pivot tables
table_fre.to_csv(table_fre_file)
table_fkg.to_csv(table_fkg_file)
