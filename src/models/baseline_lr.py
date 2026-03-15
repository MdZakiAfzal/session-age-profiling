import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def run_experiment(csv_path, text_column):
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

    print("\nResults for:", csv_path)
    print(classification_report(y_test, predictions))


print("Running tweet experiment...")
run_experiment("data/processed/tweets.csv", "text")

print("\nRunning session experiment...")
run_experiment("data/processed/sessions.csv", "text")