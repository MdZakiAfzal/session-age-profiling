def load_labels(truth_file):

    labels = {}

    with open(truth_file, "r") as f:
        for line in f:
            parts = line.strip().split(":::")
            author_id = parts[0]
            age = parts[2]

            labels[author_id] = age

    return labels