# Session-Based Age Prediction from Social Media Text

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![License](https://img.shields.io/badge/License-Research-green)

Predicting demographic attributes from social media text is a challenging NLP task because individual posts often contain limited stylistic information.

This project investigates whether **aggregating multiple tweets into sessions improves age prediction performance** compared to single tweet classification.

Several classical machine learning and transformer-based models are evaluated using the **PAN Author Profiling dataset**.

---

# Project Overview

The central research question explored in this project is:

> **Does aggregating multiple tweets into sessions improve demographic prediction compared to single tweet classification?**

Two input representations are studied:

* **Single Tweet**
* **Tweet Session (multiple tweets concatenated)**

The hypothesis is that **sessions capture richer stylistic and lexical signals**, enabling models to better infer demographic attributes.

---

# Dataset

This project uses the **PAN Author Profiling dataset**, a benchmark dataset for demographic inference research.

Each author contains approximately **100 tweets** along with demographic labels.

### Age Groups

* 18–24
* 25–34
* 35–49
* 50+

---

# Input Representations

## Single Tweet

Each tweet is treated as an independent training sample.

Example:

```
good morning everyone!
```

---

## Tweet Session

Five tweets from the same author are aggregated to form a session.

```
tweet1 [SEP] tweet2 [SEP] tweet3 [SEP] tweet4 [SEP] tweet5
```

This representation provides stronger signals for:

* vocabulary usage
* punctuation patterns
* emoji usage
* stylistic writing behavior

---

# Models Evaluated

| Model                                 | Representation | Description                           |
| ------------------------------------- | -------------- | ------------------------------------- |
| TF-IDF + Logistic Regression          | Tweet          | Lexical baseline                      |
| TF-IDF + Logistic Regression          | Session        | Context aggregation baseline          |
| BERT Embeddings + Logistic Regression | Session        | Frozen transformer embeddings         |
| BERT Fine-tuned                       | Tweet          | Transformer trained on single tweets  |
| BERT Fine-tuned                       | Session        | Transformer trained on tweet sessions |

---

# Main Results

| Model                | Input   | Accuracy | Macro F1 |
| -------------------- | ------- | -------- | -------- |
| TF-IDF + LR          | Tweet   | 0.64     | 0.57     |
| TF-IDF + LR          | Session | 0.78     | 0.73     |
| BERT Embeddings + LR | Session | 0.67     | 0.60     |
| BERT Fine-tuned      | Tweet   | 0.71     | 0.65     |
| BERT Fine-tuned      | Session | **0.86** | **0.83** |

---

# Session Size Ablation

To analyze the effect of contextual aggregation, we conducted an ablation study by varying the number of tweets per session.

| Session Size | Accuracy  | Macro F1  |
| ------------ | --------- | --------- |
| 3 Tweets     | 0.809     | 0.743     |
| 5 Tweets     | 0.860     | **0.825** |
| 10 Tweets    | **0.880** | 0.814     |

These results show that **increasing session size improves prediction accuracy**, as more tweets provide richer stylistic context. However, performance gains diminish beyond a certain session length.

---

# Ablation Visualization

![Session Ablation](figures/session_ablation_plot.png)

The figure shows that performance improves significantly when moving from 3 tweets to 5 tweets per session, while further increases provide smaller gains.

---

# Confusion Matrix (Best Model)

![Confusion Matrix](figures/bert_session_confusion_matrix.png)

The confusion matrix illustrates the performance of the **fine-tuned BERT session model**, which achieved the best results.

Most misclassifications occur between **adjacent age groups**, suggesting overlapping linguistic patterns between neighboring demographics.

---

# Key Findings

1. Aggregating tweets into sessions significantly improves age prediction performance.

2. Fine-tuned BERT models outperform classical lexical approaches.

3. Frozen transformer embeddings perform worse than TF-IDF baselines.

4. Increasing session size improves performance but shows diminishing returns.

---

# Project Structure

```
session-age-profiling

data/
   raw/
   processed/

experiments/
   (stores experiment logs and additional runs)

results/
   final_results.csv
   session_ablation_results.csv

figures/
   model_comparison.png
   bert_session_confusion_matrix.png
   session_ablation_plot.png

src/
   models/
      baseline_lr.py
      bert_embeddings.py
      bert_finetune.py

   analysis/
      plot_results.py

   run_all_experiments.py

notebooks/
   bert_experiments.ipynb

requirements.txt
README.md
```

---

# Installation

Clone the repository:

```
git clone https://github.com/MdZakiAfzal/session-age-profiling.git
cd session-age-profiling
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Running Experiments

Run the full experiment pipeline:

```
python src/run_all_experiments.py
```

This script runs:

* TF-IDF baseline models
* BERT embedding baseline
* BERT fine-tuning on single tweets
* BERT fine-tuning on tweet sessions

---

# Notebook

The notebook demonstrates the core experiment comparing tweet-based and session-based BERT models.

```
notebooks/bert_experiments.ipynb
```

It includes:

* dataset preparation
* model training
* evaluation
* visualization of results

---

# Reproducibility

All experiments are reproducible using the scripts inside `src/`.

Key features:

* deterministic train/test splits
* consistent preprocessing pipeline
* reusable experiment scripts

---

# Future Work

Potential extensions of this research include:

* incorporating user metadata
* experimenting with larger transformer architectures
* multilingual author profiling
* predicting additional demographic attributes

---

# Citation

If you use this work, please cite:

```
@misc{session_age_profiling,
  title={Session-Based Age Prediction from Social Media Text},
  author={Md Zaki Afzal},
  year={2026},
  url={https://github.com/MdZakiAfzal/session-age-profiling}
}
```

---

# Author

**Md Zaki Afzal**
Machine Learning Engineer
