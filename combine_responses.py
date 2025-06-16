import pandas as pd

# Paths to individual model files
files = [
    "Generated_Responses_ChatGPT.csv",
    "Generated_Responses_Claude.csv",
    "Generated_Responses_Gemini.csv"
]

# Load and concatenate
dfs = [pd.read_csv(f) for f in files]
combined_df = pd.concat(dfs, ignore_index=True)

# Ensure no missing critical fields
required_columns = ["Model", "Age", "Topic", "Repeat", "Prompt", "Response"]
assert all(col in combined_df.columns for col in required_columns), "Missing expected columns"

# Save to unified file
combined_df.to_csv("All_Model_Responses.csv", index=False)
