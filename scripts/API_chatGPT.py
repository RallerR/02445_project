import openai
import pandas as pd
import time
import os

# API key
client = openai.OpenAI(api_key="")

# Parameters
ages = [10, 25, 50]
n_repeats = 1
model = "gpt-4o"
temperature = 0.7
batch_size = 10

# Construct paths relative to this script
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "..", "data")
topics_csv_file = os.path.join(data_dir, "topics_list.csv")
output_file = os.path.join(data_dir, "Generated_Responses_ChatGPT.csv")

# Load topics from list
topic_df = pd.read_csv(topics_csv_file)
topics = topic_df["Topic"].dropna().unique().tolist()

# Prepare prompts
prompts = []
for topic in topics:
    for age in ages:
        for r in range(n_repeats):
            prompt = f"I am {age} years old. Can you explain {topic} to me?"
            prompts.append((age, topic, r + 1, prompt))

# Load previous responses
if os.path.exists(output_file):
    existing_df = pd.read_csv(output_file)
    completed_prompts = set(existing_df["Prompt"])
    print(f"Resuming from {len(completed_prompts)} existing prompts.")
else:
    existing_df = pd.DataFrame()
    completed_prompts = set()

# Collect responses
results = []
for i in range(0, len(prompts), batch_size):
    batch = prompts[i:i + batch_size]
    for age, topic, repeat, prompt in batch:
        if prompt in completed_prompts:
            continue
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            text = response.choices[0].message.content
            results.append({
                "Model": model,
                "Age": age,
                "Topic": topic,
                "Repeat": repeat,
                "Prompt": prompt,
                "Response": text
            })
            print(f"Done: {prompt[:40]}...")
            time.sleep(1.2)
        except Exception as e:
            print(f"Error for: {prompt[:40]} → {e}")
            time.sleep(3)

    # Save after each batch
    if results:
        batch_df = pd.DataFrame(results)
        all_df = pd.concat([existing_df, batch_df], ignore_index=True)
        all_df.to_csv(output_file, index=False)
        existing_df = all_df
        results = []
        print(f"Batch saved ({len(all_df)} total rows)\n")
