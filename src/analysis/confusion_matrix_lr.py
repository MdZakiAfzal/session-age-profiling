import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import os

def run_experiment(csv_path, text_column, output_name):

    df = pd.read_csv(csv_path)
    df["age_label"] = df["age_label"].replace("50-XX", "50+")

    X_train, X_test, y_train, y_test = train_test_split(
        df[text_column],
        df["age_label"],
        test_size=0.2,
        random_state=42,
        stratify=df["age_label"]
    )

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1,2)
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)

    cm = confusion_matrix(y_test, predictions, labels=model.classes_)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=model.classes_,
        yticklabels=model.classes_,
        cmap="Blues"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(output_name)

    os.makedirs("figures", exist_ok=True)

    plt.savefig(f"figures/{output_name}.png")
    plt.close()

    print(f"Saved: figures/{output_name}.png")

run_experiment(
    "data/processed/tweets.csv",
    "tweet_text",
    "tweet_confusion_matrix"
)

run_experiment(
    "data/processed/sessions.csv",
    "session_text",
    "session_confusion_matrix"
)