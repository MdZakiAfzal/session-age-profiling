from src.utils.text_cleaning import clean_tweet

def build_sessions(authors, labels, session_size=5):

    sessions = []
    session_labels = []

    for author_id, tweets in authors.items():

        age = labels[author_id]

        for i in range(0, len(tweets), session_size):

            chunk = tweets[i:i+session_size]

            if len(chunk) == session_size:

                chunk = [clean_tweet(t) for t in chunk]
                session_text = " [SEP] ".join(chunk)

                sessions.append(session_text)
                session_labels.append(age)

    return sessions, session_labels