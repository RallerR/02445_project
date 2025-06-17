import pandas as pd
import glob
import os

# Set base and data directories
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "..", "data")

# Load all evaluation CSV files (e.g., evaluations_user1.csv, ...)
file_pattern = os.path.join(data_dir, "evaluations_user*.csv")
user_files = glob.glob(file_pattern)

all_user_data = []

for file in user_files:
    df = pd.read_csv(file)
    user_id = file.split("_user")[1].split(".")[0]  # extract user identifier from filename
    df["User"] = f"user{user_id}"
    # Compute Fluency score
    df["Fluency"] = df[["Vocabulary", "Tone", "Analogy", "Clarity"]].mean(axis=1)
    all_user_data.append(df)

# Combine all user data
df_all = pd.concat(all_user_data, ignore_index=True)

# Save full individual ratings with Fluency added
all_eval_path = os.path.join(data_dir, "All_Evaluations_With_Fluency.csv")
df_all.to_csv(all_eval_path, index=False)

# Compute averaged fluency per sample
group_keys = ["Prompt", "Model", "Age", "Repeat", "Topic"]

# First occurrence of Response
response_df = df_all.groupby(group_keys, as_index=False)["Response"].first()

# Mean of all relevant metrics
metrics_to_avg = ["Vocabulary", "Tone", "Analogy", "Clarity", "Fluency"]
averaged_df = df_all.groupby(group_keys)[metrics_to_avg].mean().reset_index()

# Merge to restore Response column
averaged_with_response = pd.merge(averaged_df, response_df, on=group_keys)

# Reorder columns
column_order = [
    "Model", "Age", "Topic", "Repeat", "Prompt",
    "Response",
    "Vocabulary", "Tone", "Analogy", "Clarity", "Fluency"
]
averaged_with_response = averaged_with_response[column_order]

# Save averaged scores
avg_eval_path = os.path.join(data_dir, "Averaged_Evaluations.csv")
averaged_with_response.to_csv(avg_eval_path, index=False)
