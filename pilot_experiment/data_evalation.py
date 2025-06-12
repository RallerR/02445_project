import pandas as pd

def get_valid_score(prompt):
    while True:
        try:
            score = int(input(prompt))
            if 1 <= score <= 5:
                return score
            else:
                print("Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 5.")

# Load the CSV file with semicolon
df = pd.read_csv("PilotData.csv", delimiter=';', on_bad_lines='skip')

# Initialize lists to store evaluation scores
vocabulary_scores = []
tone_scores = []
analogy_scores = []
clarity_scores = []

# Iterate through each row and collect user evaluations
for index, row in df.iterrows():
    print(f"\n--- Evaluation {index + 1} of {len(df)} ---")
    print(f"Model: {row['Model']}")
    print(f"Concept: {row['Concept']}")
    print(f"Age: {row['Age']}")
    print(f"Prompt: {row['Prompt']}")
    print(f"Response: {row['Response']}\n")

    # Get validated user input for each metric
    vocabulary = get_valid_score("Score for Vocabulary (1-5): ")
    tones = get_valid_score("Score for Tone (1-5): ")
    analogy = get_valid_score("Score for Analogy (1-5): ")
    clarity = get_valid_score("Score for Clarity (1-5): ")

    # Append scores to respective lists
    vocabulary_scores.append(vocabulary)
    tone_scores.append(tones)
    analogy_scores.append(analogy)
    clarity_scores.append(clarity)

# Add the scores to the dataframe
df["Vocabulary"] = vocabulary_scores
df["Tone"] = tone_scores
df["Analogy"] = analogy_scores
df["Clarity"] = clarity_scores

# Drop empty rows (which arent supposed to be there
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.dropna(axis=1, how='all')


# Save the updated dataframe to a new CSV file
df.to_csv("PilotData_Evaluated.csv", index=False)

print("\nEvaluation complete. Scores saved to 'PilotData_Evaluated.csv'.")
