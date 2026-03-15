import pandas as pd

from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def run_experiment(csv_path):

    df = pd.read_csv(csv_path)
    df["age_label"] = df["age_label"].replace("50-XX", "50+")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["age_label"],
        test_size=0.2,
        random_state=42,
        stratify=df["age_label"]
    )

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Generating embeddings...")

    X_train_embeddings = model.encode(
        X_train.tolist(),
        show_progress_bar=True
    )

    X_test_embeddings = model.encode(
        X_test.tolist(),
        show_progress_bar=True
    )

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    clf.fit(X_train_embeddings, y_train)

    predictions = clf.predict(X_test_embeddings)

    print(classification_report(y_test, predictions))


if __name__ == "__main__":

    run_experiment("../../data/processed/bert_sessions.csv")