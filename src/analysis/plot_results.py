import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/final_results.csv")

plt.figure(figsize=(8,5))

plt.bar(
    df["model"] + "_" + df["input"],
    df["accuracy"]
)

plt.ylabel("Accuracy")
plt.title("Model Comparison")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("figures/model_comparison.png")

print("Saved figures/model_comparison.png")