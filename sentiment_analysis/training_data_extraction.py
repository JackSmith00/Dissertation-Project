"""
Used to extract articles from the csv of a labelled dataset containing
news articles labelled as biased/not biased/inconclusive. Each article
is extracted to its own txt file inside a folder with the appropriate
label. The labelled dataset used is referenced below:

Spinde, T., 2021. MBIC – A Media Bias Annotation Dataset [online].
Available from: https://www.kaggle.com/datasets/timospinde/mbic-a-media-bias-annotation-dataset?resource=download
[Accessed 19 April 2023].
"""
import pandas as pd
from nltk.corpus import stopwords

# read in the csv to a pandas dataframe to easily access data
labelled_dataset = pd.read_csv("/Volumes/24265241/Supervised Training/labeled_dataset.csv")


def extract_biased_words(save_path: str):
    """
    Used to extract the all words listed in the 'biased_words4' column
    of the labelled dataset. These are words that were identified as
    indicating bias when the dataset was originally labelled.
    :param save_path: location to save the words to
    """
    biased_words = labelled_dataset["biased_words4"].values

    output_holder = set()  # prevents duplicates being extracted

    for word_list in biased_words:
        if word_list == "[]":  # ignore entries with no words
            continue

        list_items = word_list.strip("[]").split(", ")  # remove brackets and split by comma
        for item in list_items:
            output_holder.add(item.strip("'\"").lower())  # remove quote marks and add to the word list

    # save to external file
    with open(save_path, "w") as f:
        for item in sorted(output_holder):  # sort the words for ease of use in future
            if item.casefold() != "" and item not in stopwords.words("english"):  # ignore any blank lines or stopwords
                f.write(item + "\n")  # save each word on a new line


def extract_training_data():
    """
    Used to extract all articles from the labelled dataset to an appropriate
    pre-defined folder based on their assigned label in order to be used for
    training a supervised model to detect the difference between biased and
    non-biased news articles. Any articles that have not been classified as
    either biased or non-biased are also extracted to a separate folder as
    features in these articles are seemingly inconclusive and should not be
    considered as features for the classification model.
    """
    required_data = labelled_dataset[["Label_bias", "article"]].values

    # count articles as to not overwrite articles already extracted
    biased_article_count = 0
    non_biased_article_count = 0
    unclassified_article_count = 0

    for article in required_data:
        if article[1].__class__ != str:
            # handle entries with no article text
            continue
        if article[0] == "Biased":
            # save biased articles to a biased folder
            with open(f"/Volumes/24265241/Supervised Training/Biased/article{biased_article_count}.txt", "w") as f:
                f.write(article[1])
                biased_article_count += 1
        elif article[0] == "Non-biased":
            # save non-biased articles to a non-biased folder
            with open(f"/Volumes/24265241/Supervised Training/Non-biased/article{non_biased_article_count}.txt",
                      "w") as f:
                f.write(article[1])
                non_biased_article_count += 1
        else:
            # save unclassified articles to a separate folder
            with open(f"/Volumes/24265241/Supervised Training/Unclassified/article{unclassified_article_count}.txt",
                      "w") as f:
                f.write(article[1])
                unclassified_article_count += 1


if __name__ == '__main__':

    extract_biased_words("/Volumes/24265241/Supervised Training/biased_words.txt")
    extract_training_data()
