import pandas as pd
import os

# Define data directory path relative to the script
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "..", "data")

# Paths to individual model files in the data folder
files = [
    os.path.join(data_dir, "Generated_Responses_ChatGPT.csv"),
    os.path.join(data_dir, "Generated_Responses_Claude.csv"),
    os.path.join(data_dir, "Generated_Responses_Gemini.csv")
]

# Load and concatenate
dfs = [pd.read_csv(f) for f in files]
combined_df = pd.concat(dfs, ignore_index=True)

# Ensure no missing critical fields
required_columns = ["Model", "Age", "Topic", "Repeat", "Prompt", "Response"]
assert all(col in combined_df.columns for col in required_columns), "Missing expected columns"

# Save to unified file in data folder
output_file = os.path.join(data_dir, "All_Model_Responses.csv")
combined_df.to_csv(output_file, index=False)