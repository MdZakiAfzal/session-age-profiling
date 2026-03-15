import json
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from src.utils.metrics import compute_metrics


def train_bert(csv_path):

    df = pd.read_csv(csv_path)

    df["age_label"] = df["age_label"].replace("50-XX", "50+")

    labels = sorted(df["age_label"].unique())

    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    df["label"] = df["age_label"].map(label2id)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    model_name = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(example):

        return tokenizer(
            example["text"],
            truncation=True,
            max_length=256
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=6,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    results = trainer.evaluate()

    with open("results/bert_results.json", "w") as f:
        json.dump(results, f)
    print(results)


import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset csv"
    )

    args = parser.parse_args()

    train_bert(args.dataset)