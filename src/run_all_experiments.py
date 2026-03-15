import subprocess


def run_command(cmd):

    print("\n==============================")
    print("Running:", cmd)
    print("==============================\n")

    subprocess.run(cmd, shell=True, check=True)


def main():

    # 1️⃣ TF-IDF baseline
    run_command(
        "python src/models/baseline_lr.py"
    )

    # 2️⃣ BERT embeddings baseline
    run_command(
        "python src/models/bert_embeddings.py"
    )

    # 3️⃣ BERT fine-tuning (tweet model)
    run_command(
        "python src/models/bert_finetune.py --dataset data/processed/bert_tweets.csv"
    )

    # 4️⃣ BERT fine-tuning (session model)
    run_command(
        "python src/models/bert_finetune.py --dataset data/processed/bert_sessions.csv"
    )

    print("\nAll experiments completed!")


if __name__ == "__main__":

    main()