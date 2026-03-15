from src.utils.text_cleaning import clean_tweet

def build_tweet_dataset(authors, labels):

    tweets = []
    tweet_labels = []

    for author_id, author_tweets in authors.items():
        age = labels[author_id]
        for tweet in author_tweets:
            tweet = clean_tweet(tweet)
            if tweet != "":
                tweets.append(tweet)
                tweet_labels.append(age)

    return tweets, tweet_labels