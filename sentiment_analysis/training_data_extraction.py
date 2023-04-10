import pandas as pd
from nltk.corpus import stopwords

labeled_dataset = pd.read_csv("/Volumes/24265241/Unsupervised Training/labeled_dataset.csv")


def extract_biased_words(save_path: str):
    biased_words = labeled_dataset["biased_words4"].values

    output_holder = set()

    for word_list in biased_words:
        if word_list == "[]":
            continue

        list_items = word_list.strip("[]").split(", ")
        for item in list_items:
            output_holder.add(item.strip("'\"").lower())

    with open(save_path, "w") as f:
        for item in sorted(output_holder):
            if item.casefold() != "" and item not in stopwords.words("english"):
                f.write(item + "\n")


def extract_training_data():
    required_data = labeled_dataset[["Label_bias", "article"]].values

    biased_article_count = 0
    non_biased_article_count = 0
    unclassified_article_count = 0

    for article in required_data:
        if article[1].__class__ != str:
            continue
        if article[0] == "Biased":
            with open(f"/Volumes/24265241/Unsupervised Training/Biased/article{biased_article_count}.txt", "w") as f:
                f.write(article[1])
                biased_article_count += 1
        elif article[0] == "Non-biased":
            with open(f"/Volumes/24265241/Unsupervised Training/Non-biased/article{non_biased_article_count}.txt",
                      "w") as f:
                f.write(article[1])
                non_biased_article_count += 1
        else:
            with open(f"/Volumes/24265241/Unsupervised Training/Unclassified/article{unclassified_article_count}.txt",
                      "w") as f:
                f.write(article[1])
                unclassified_article_count += 1


if __name__ == '__main__':

    extract_biased_words("/Volumes/24265241/Unsupervised Training/biased_words.txt")
    extract_training_data()
