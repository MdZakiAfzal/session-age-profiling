import pandas as pd

from src.data.build_tweet_dataset import build_tweet_dataset

from src.data.load_authors import load_authors
from src.data.load_labels import load_labels
from src.data.build_sessions import build_sessions

dataset_path = "data/raw/pan15"
truth_file = "data/raw/pan15/truth.txt"

authors = load_authors(dataset_path)
labels = load_labels(truth_file)

sessions, session_labels = build_sessions(authors, labels, session_size=5)

print("Total sessions:", len(sessions))
print("Example:", sessions[0])
print("Label:", session_labels[0])

# ---- Save dataset ----

df = pd.DataFrame({
    "session_text": sessions,
    "age_label": session_labels
})

output_path = "data/processed/sessions_5.csv"

df.to_csv(output_path, index=False)

print("Saved dataset to:", output_path)
print("Dataset shape:", df.shape)

tweets, tweet_labels = build_tweet_dataset(authors, labels)

tweet_df = pd.DataFrame({
    "tweet_text": tweets,
    "age_label": tweet_labels
})

tweet_output = "data/processed/tweets.csv"

tweet_df.to_csv(tweet_output, index=False)

print("Saved tweet dataset:", tweet_output)
print("Tweet dataset size:", tweet_df.shape)
# total 14,166 tweets in 2852 sessions, 4 age groups (18-24, 25-34, 35-49, 50+)