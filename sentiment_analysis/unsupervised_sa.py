from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import PlaintextCorpusReader
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy, SklearnClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from textblob import TextBlob
from flair.models import TextClassifier
from flair.data import Sentence
from statistics import mean, mode
from random import shuffle
from tqdm import tqdm
from tabulate import tabulate
import pickle
import os.path

vader_analyser = SentimentIntensityAnalyzer()
flair_analyser = TextClassifier.load("en-sentiment")

biased_words = []
manually_extracted_biased_words = []
non_biased_words = []
subjective_words = []
objective_words = []

biased_bigrams = []
non_biased_bigrams = []
subjective_bigrams = []
objective_bigrams = []


def load_feature(feature: list, location: str):
    with open(location) as f:
        for line in f.readlines():
            feature.append(line.strip())


def load_all_features():
    load_feature(biased_words, "features/biased_words.txt")
    load_feature(manually_extracted_biased_words, "/Volumes/24265241/Unsupervised Training/biased_words.txt")
    load_feature(non_biased_words, "features/non_biased_words.txt")
    load_feature(subjective_words, "features/subjective_words.txt")
    load_feature(objective_words, "features/objective_words.txt")
    load_feature(biased_bigrams, "features/biased_bigrams.txt")
    load_feature(non_biased_bigrams, "features/non_biased_bigrams.txt")
    load_feature(subjective_bigrams, "features/subjective_bigrams.txt")
    load_feature(objective_bigrams, "features/objective_bigrams.txt")


def extract_features(text: str) -> dict:
    # todo: ref sent analysis real python
    # todo: try all features and then remove to see effect on accuracy
    extracted_features = dict()
    biased_word_count = 0
    man_biased_word_count = 0
    non_biased_word_count = 0
    subjective_word_count = 0
    objective_word_count = 0
    biased_bigram_count = 0
    non_biased_bigram_count = 0
    subjective_bigram_count = 0
    objective_bigram_count = 0
    compound_vader_scores = []
    textblob_scores = []
    flair_tags = []
    flair_confidence = []

    for sentence in sent_tokenize(text):
        for word in word_tokenize(sentence):
            if word.casefold() in biased_words:
                biased_word_count += 1
            if word.casefold() in manually_extracted_biased_words:
                man_biased_word_count += 1
            if word.casefold() in non_biased_words:
                non_biased_word_count += 1
            if word.casefold() in subjective_words:
                subjective_word_count += 1
            if word.casefold() in objective_words:
                objective_word_count += 1
        for bigram in biased_bigrams:
            biased_bigram_count += sentence.count(bigram)
        for bigram in non_biased_bigrams:
            non_biased_bigram_count += sentence.count(bigram)
        for bigram in subjective_bigrams:
            subjective_bigram_count += sentence.count(bigram)
        for bigram in objective_bigrams:
            objective_bigram_count += sentence.count(bigram)

        compound_vader_scores.append(vader_analyser.polarity_scores(sentence)["compound"])
        textblob_scores.append(TextBlob(sentence).subjectivity)

        flair_sent = Sentence(sentence)
        flair_analyser.predict(flair_sent)
        flair_tags.append(flair_sent.tag)
        flair_confidence.append(flair_sent.score)

    extracted_features["biased_word_count"] = biased_word_count
    extracted_features["manual_biased_word_count"] = man_biased_word_count
    extracted_features["non_biased_word_count"] = non_biased_word_count
    extracted_features["subjective_word_count"] = subjective_word_count
    extracted_features["objective_word_count"] = objective_word_count
    extracted_features["biased_bigram_count"] = biased_bigram_count
    extracted_features["non_biased_bigram_count"] = non_biased_bigram_count
    extracted_features["subjective_bigram_count"] = subjective_bigram_count
    extracted_features["objective_bigram_count"] = objective_bigram_count
    extracted_features["vader_score"] = mean(compound_vader_scores) + 1  # some classifiers require positive numbers
    extracted_features["textblob_score"] = mean(textblob_scores)
    extracted_features["flair_tag"] = 1 if mode(flair_tags) == "POSITIVE" else 0
    extracted_features["flair_percentage"] = flair_tags.count(mode(flair_tags)) / len(flair_tags)
    extracted_features["flair_confidence"] = mode(flair_confidence)

    return extracted_features


if __name__ == '__main__':

    corpora_regex = "[a-zA-Z0-9-_]*.txt"

    # read feature word lists
    #load_all_features()

    # get features for training
    if os.path.exists("/Volumes/24265241/Unsupervised Training/labeled_dataset_features.pkl"):
        with open("/Volumes/24265241/Unsupervised Training/labeled_dataset_features.pkl", "rb") as f:
            labeled_dataset_features = pickle.load(f)
    else:
        biased_articles = PlaintextCorpusReader("/Volumes/24265241/Unsupervised Training/Biased", corpora_regex)
        non_biased_articles = PlaintextCorpusReader("/Volumes/24265241/Unsupervised Training/Non-biased", corpora_regex)

        labeled_dataset_features = [
            (extract_features(biased_articles.raw(article)), 'biased')
            for article in tqdm(biased_articles.fileids(), desc="Biased feature extraction")
        ]

        labeled_dataset_features.extend([
            (extract_features(non_biased_articles.raw(article)), 'not biased')
            for article in tqdm(non_biased_articles.fileids(), desc="Non-biased feature extraction")
        ])

        with open("/Volumes/24265241/Unsupervised Training/labeled_dataset_features.pkl", "wb") as f:
            pickle.dump(labeled_dataset_features, f)

    # train various classifiers
    train_count = (len(labeled_dataset_features) // 4) * 3  # train with 75% of the dataset

    for i in range(1, 11):
        shuffle(labeled_dataset_features)
        model_accuracies = dict()

        # handle naive bayes separately as from nltk not sklearn
        naive_bayes = NaiveBayesClassifier.train(labeled_dataset_features[:train_count])
        model_accuracies["Naive Bayes"] = accuracy(naive_bayes, labeled_dataset_features[train_count:])

        # handle remaining classifiers
        sklearn_classifiers = {
            "BernoulliNB": BernoulliNB(),
            "ComplementNB": ComplementNB(),
            "MultinomialNB": MultinomialNB(),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "RandomForestClassifier": RandomForestClassifier(),
            "LogisticRegression": LogisticRegression(max_iter=1500),
            "MLPClassifier": MLPClassifier(max_iter=1500),
            "AdaBoostClassifier": AdaBoostClassifier(),
        }

        for model_name, model in tqdm(sklearn_classifiers.items(), desc="Testing Models"):
            classifier = SklearnClassifier(model)
            classifier.train(labeled_dataset_features[:train_count])
            model_accuracies[model_name] = accuracy(classifier, labeled_dataset_features[train_count:])

        print(tabulate(model_accuracies.items(), headers=["Model", "Accuracy"], floatfmt=".2%"))

        with open("/Volumes/24265241/Unsupervised Training/model_accuracies.txt", "a") as f:
            f.write(f"Test {i}\n------\n")
            f.write(tabulate(model_accuracies.items(), headers=["Model", "Accuracy"]) + "\n\n")
