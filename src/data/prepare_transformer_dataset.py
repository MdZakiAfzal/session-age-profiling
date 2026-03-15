import pandas as pd

sessions = pd.read_csv("data/processed/sessions.csv")
tweets = pd.read_csv("data/processed/tweets.csv")

sessions["age_label"] = sessions["age_label"].replace("50-XX", "50+")
tweets["age_label"] = tweets["age_label"].replace("50-XX", "50+")

sessions = sessions.rename(columns={"session_text": "text"})
tweets = tweets.rename(columns={"tweet_text": "text"})

sessions.to_csv("data/processed/bert_sessions.csv", index=False)
tweets.to_csv("data/processed/bert_tweets.csv", index=False)

print("Saved BERT datasets.")
print("Sessions shape:", sessions.shape)
print("Tweets shape:", tweets.shape)