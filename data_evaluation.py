import pandas as pd
import os
import random
import re

# Set the evaluator's ID
USER_ID = "user1"

INPUT_FILE = "All_Model_Responses.csv"
OUTPUT_FILE = f"evaluations_{USER_ID}.csv"

def clean_markdown(text):
    text = re.sub(r"(\*\*|\*|_)(.*?)\1", r"\2", text)  # remove markdown
    text = re.sub(r"\$.*?\$", "[math]", text)          # remove LaTeX-style math
    return text

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

# --- Show introduction ---
print("In this experiment, you'll be shown a short prompt and an AI-generated response, base on different ages an topics.")
print("Your task is to evaluate the quality of the response based on **four criteria**, each on a scale from 1 (poor) to 5 (excellent):\n")

print("1. Vocabulary: Does the language suit the target age group? Are words understandable in regards to the target age group?")
print("2. Tone: Is the tone friendly, appropriate, and well-suited to a learner of that age?")
print("3. Analogy: Does the explanation use helpful analogies or relatable examples (if needed) suited for that age?")
print("4. Clarity: Is the explanation clear, logical, and easy to follow? Does the point come well across for a learner of that age?\n")

input("Press Enter to begin the evaluation...")

# --- Load data ---
df = pd.read_csv(INPUT_FILE)

if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
    evaluated_df = pd.read_csv(OUTPUT_FILE)
    if {"Prompt", "Age", "Repeat", "Model", "User"}.issubset(evaluated_df.columns):
        evaluated_keys = set(
            zip(
                evaluated_df["Prompt"],
                evaluated_df["Age"],
                evaluated_df["Repeat"],
                evaluated_df["Model"],
                evaluated_df["User"]
            )
        )
    else:
        print("Existing file is missing expected columns. Starting fresh.")
        evaluated_df = pd.DataFrame()
        evaluated_keys = set()
else:
    evaluated_df = pd.DataFrame()
    evaluated_keys = set()

# Filter out already evaluated samples for this user
to_evaluate = [
    row for _, row in df.iterrows()
    if (row["Prompt"], row["Age"], row["Repeat"], row["Model"], USER_ID) not in evaluated_keys
]

random.shuffle(to_evaluate)

# --- Evaluation loop ---
new_evaluations = []

for idx, row in enumerate(to_evaluate, 1):
    print(f"\n--- Evaluation {idx} of {len(to_evaluate)} ---")
    print(f"Concept: {row['Topic']}")
    print(f"Age: {row['Age']}")
    print(f"Prompt: {row['Prompt']}\n")
    print(f"Response: {clean_markdown(row['Response'])}\n")

    vocab = get_valid_score("Score for Vocabulary (1-5): ")
    tone = get_valid_score("Score for Tone (1-5): ")
    analogy = get_valid_score("Score for Analogy (1-5): ")
    clarity = get_valid_score("Score for Clarity (1-5): ")

    evaluated_row = row.to_dict()
    evaluated_row.update({
        "User": USER_ID,
        "Vocabulary": vocab,
        "Tone": tone,
        "Analogy": analogy,
        "Clarity": clarity
    })

    new_evaluations.append(evaluated_row)

    # Save after each evaluation
    evaluated_df = pd.concat([evaluated_df, pd.DataFrame([evaluated_row])], ignore_index=True)
    evaluated_df.to_csv(OUTPUT_FILE, index=False)
    print("Saved\n")

print("\nEvaluation session complete.")
