import os
import pandas as pd
import time
from google import genai
from google.genai import types

# Configure API key
client = genai.Client(api_key="AIzaSyDCeM6kxTF5fX9pK93NieaTNpF7wkIWOjE")

# Parameters
ages = [10, 25, 50]
n_repeats = 1
model_name = "models/gemini-1.5-flash"
temperature = 0.7
batch_size = 10
topics_csv_file = "topics_list.csv"
output_file = "Generated_Responses_Gemini.csv"

# Load topics
topic_df = pd.read_csv(topics_csv_file)
topics = topic_df["Topic"].dropna().unique().tolist()

# Build prompts
prompts = []
for topic in topics:
    for age in ages:
        for repeat in range(1, n_repeats + 1):
            prompt = f"I am {age} years old. Can you explain {topic} to me?"
            prompts.append((age, topic, repeat, prompt))

# Load previous responses
if os.path.exists(output_file):
    existing_df = pd.read_csv(output_file)
    done = set(existing_df["Prompt"] + "|" + existing_df["Repeat"].astype(str))
    print(f"Resuming from {len(done)} saved prompts.")
else:
    existing_df = pd.DataFrame()
    done = set()

# Collect responses
results = []
for i in range(0, len(prompts), batch_size):
    batch = prompts[i:i + batch_size]
    for age, topic, repeat, prompt_text in batch:
        key = prompt_text + "|" + str(repeat)
        if key in done:
            continue
        try:
            response = client.models.generate_content(
                model="models/gemini-1.5-flash",
                contents=prompt_text,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=512
                )
            )
            text = response.text

            results.append({
                "Model": model_name,
                "Age": age,
                "Topic": topic,
                "Repeat": repeat,
                "Prompt": prompt_text,
                "Response": text
            })
            print(f"✓ Gemini done: {prompt_text[:40]} (r{repeat})")
            time.sleep(1)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

    # Save after batch
    if results:
        batch_df = pd.DataFrame(results)
        all_df = pd.concat([existing_df, batch_df], ignore_index=True)
        all_df.to_csv(output_file, index=False)
        existing_df = all_df
        done.update(batch_df["Prompt"] + "|" + batch_df["Repeat"].astype(str))
        results = []
        print(f"Saved batch ({len(all_df)} rows)\n")
