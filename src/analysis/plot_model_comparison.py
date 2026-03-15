import matplotlib.pyplot as plt
import pandas as pd

# Final results from experiments
results = {
    "Model": [
        "TF-IDF + LR (Tweet)",
        "TF-IDF + LR (Session)",
        "BERT Embeddings + LR",
        "Fine-tuned BERT"
    ],
    "Accuracy": [0.64, 0.78, 0.67, 0.853],
    "Macro F1": [0.57, 0.73, 0.60, 0.80]
}

df = pd.DataFrame(results)

plt.figure(figsize=(8,5))

plt.bar(df["Model"], df["Accuracy"])

plt.ylabel("Accuracy")
plt.title("Model Performance Comparison")

plt.xticks(rotation=25)

plt.tight_layout()

plt.savefig("figures/model_accuracy_comparison.png")

print("Saved: figures/model_accuracy_comparison.png")