import pandas as pd

tweet_df = pd.read_csv("data/processed/tweets.csv")
session_df = pd.read_csv("data/processed/sessions.csv")

print("Tweet dataset size:", tweet_df.shape)
print(tweet_df["age_label"].value_counts())

print("\nSession dataset size:", session_df.shape)
print(session_df["age_label"].value_counts())