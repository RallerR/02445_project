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

# Save or view the result
grouped.to_csv("Aggregated_Evaluation.csv", index=False)

# Load the aggregated evaluation data
df = pd.read_csv("Aggregated_Evaluation.csv")

# Pivot table to compute mean Overall_Mean for each Model × Age
table = df.pivot_table(
    index='Age',
    columns='Model',
    values='Overall_Mean',
    aggfunc='mean'
).round(2)

# Print the table
print(table)