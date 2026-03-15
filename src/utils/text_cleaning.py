import re

def clean_tweet(text):

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)

    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    return text.strip()