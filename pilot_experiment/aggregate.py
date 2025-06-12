import pandas as pd

# Load your CSV
df = pd.read_csv("PilotData_Evaluated.csv")

# Group by Model, Concept, and Age
grouped = df.groupby(['Model', 'Concept', 'Age']).agg(
    Vocabulary_Mean=('Vocabulary', 'mean'),
    Tone_Mean=('Tone', 'mean'),
    Analogy_Mean=('Analogy', 'mean'),
    Clarity_Mean=('Clarity', 'mean'),

    Vocabulary_Std=('Vocabulary', 'std'),
    Tone_Std=('Tone', 'std'),
    Analogy_Std=('Analogy', 'std'),
    Clarity_Std=('Clarity', 'std')
).reset_index()

# Add overall score (mean of all four metrics)
grouped['Overall_Mean'] = grouped[['Vocabulary_Mean', 'Tone_Mean', 'Analogy_Mean', 'Clarity_Mean']].mean(axis=1)
grouped['Overall_Std'] = grouped[['Vocabulary_Std', 'Tone_Std', 'Analogy_Std', 'Clarity_Std']].mean(axis=1)

# Save to CSV if needed
grouped.to_csv("Aggregated_Evaluation.csv", index=False)

# Now create a summary table with both Mean and Std of the Overall scores per Model × Age
summary = grouped.groupby(['Model', 'Age']).agg(
    Mean_Overall=('Overall_Mean', 'mean'),
    Std_Overall=('Overall_Mean', 'std')
).reset_index()

# Pivot table to show both stats side-by-side
summary_table = summary.pivot(index='Age', columns='Model')

# Clean up column names for readability
summary_table.columns = ['_'.join(col).strip() for col in summary_table.columns.values]

# Round values for clarity
summary_table = summary_table.round(2)

# Print or save
print(summary_table)
summary_table.to_csv("Overall_Stats_By_Model_Age.csv")