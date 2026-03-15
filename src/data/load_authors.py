import os
import xml.etree.ElementTree as ET

def load_authors(dataset_path):

    authors = {}

    for file in os.listdir(dataset_path):

        if file.endswith(".xml"):

            author_id = file.replace(".xml", "")
            file_path = os.path.join(dataset_path, file)

            tree = ET.parse(file_path)
            root = tree.getroot()

            tweets = []

            for doc in root.iter("document"):
                tweets.append(doc.text)

            authors[author_id] = tweets

    return authors