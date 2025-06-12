import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the aggregated data
df = pd.read_csv("Aggregated_Evaluation.csv")

# Create box plot: Overall Mean score by Model and Age
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Age', y='Overall_Mean', hue='Model')

# Styling
plt.title("Box Plot of Fluency Scores")
plt.ylabel("Fluency Score (1–5)")
plt.xlabel("Age Group")
plt.ylim(0.5, 5.5)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(title='Model')
plt.tight_layout()

# Show the plot
plt.show()
