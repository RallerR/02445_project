import anthropic
import pandas as pd
import time
import os

# API key
client = anthropic.Anthropic(api_key="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")

# Parameters (same as your ChatGPT setup)
ages = [10, 25, 50]
n_repeats = 1
model = "claude-3-7-sonnet-20250219"  # choose the Claude version you want
temperature = 0.7
batch_size = 10
topics_csv_file = "topics_list.csv"
output_file = "Generated_Responses_Claude.csv"

# Load topics
topic_df = pd.read_csv(topics_csv_file)
topics = topic_df["Topic"].dropna().unique().tolist()

# Build prompt list
prompts = []
for topic in topics:
    for age in ages:
        for r in range(1, n_repeats + 1):
            prompt_text = f"I am {age} years old. Can you explain {topic} to me?"
            prompts.append((age, topic, r, prompt_text))

# Load saved responses if existing
if os.path.exists(output_file):
    existing_df = pd.read_csv(output_file)
    done = set(existing_df["Prompt"] + existing_df["Repeat"].astype(str))
    print(f"Resuming from {len(done)} saved prompt/repeat combos.")
else:
    existing_df = pd.DataFrame()
    done = set()

results = []

# Process in batches
for i in range(0, len(prompts), batch_size):
    batch = prompts[i:i + batch_size]
    for age, topic, repeat, prompt_text in batch:
        key = prompt_text + f"|{repeat}"
        if key in done:
            continue
        try:
            resp = client.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=temperature,
                max_tokens=512
            )
            answer = resp.content[0].text
            results.append({
                "Model": model,
                "Age": age,
                "Topic": topic,
                "Repeat": repeat,
                "Prompt": prompt_text,
                "Response": answer
            })
            print(f"✓ Claude done: {prompt_text[:40]} (r{repeat})")
            time.sleep(1)  # avoid rate limits
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(3)

    # Save after each batch
    if results:
        batch_df = pd.DataFrame(results)
        all_df = pd.concat([existing_df, batch_df], ignore_index=True)
        all_df.to_csv(output_file, index=False)
        existing_df = all_df
        done.update(batch_df["Prompt"] + '|' + batch_df["Repeat"].astype(str))
        results = []
        print(f"Saved batch ({len(all_df)} rows)\n")
