import os
import pandas as pd
import textstat

# Resolve correct paths
project_root = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(project_root, "data")
os.makedirs(data_dir, exist_ok=True)

input_file = os.path.join(data_dir, "Averaged_Evaluations.csv")
output_file = os.path.join(data_dir, "Averaged_Evaluations_with_Readability.csv")
table_fre_file = os.path.join(data_dir, "Flesch_Reading_Ease_by_model_age.csv")
table_fkg_file = os.path.join(data_dir, "Flesch_Kincaid_Grade_by_model_age.csv")

# Load the averaged evaluations
df = pd.read_csv(input_file)

# Compute readability scores
df["Flesch_Reading_Ease"] = df["Response"].apply(textstat.flesch_reading_ease)
df["Flesch_Kincaid_Grade"] = df["Response"].apply(textstat.flesch_kincaid_grade)

# Save enriched data
df.to_csv(output_file, index=False)
print(f"Saved enriched data with readability scores to: {output_file}")

# Group by Age and Model and compute mean and std for FRE
fre_stats = df.groupby(["Model", "Age"])["Flesch_Reading_Ease"].agg(["mean", "std"]).round(2).reset_index()
fre_stats.to_csv(table_fre_file, index=False)
print(f"Saved FRE stats to: {table_fre_file}")

# Group by Age and Model and compute mean and std for FKG
fkg_stats = df.groupby(["Model", "Age"])["Flesch_Kincaid_Grade"].agg(["mean", "std"]).round(2).reset_index()
fkg_stats.to_csv(table_fkg_file, index=False)
print(f"Saved FKG stats to: {table_fkg_file}")
