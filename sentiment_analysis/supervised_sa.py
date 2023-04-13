"""
Used to analyse the sentiment of news articles using a supervised approach
and give each article a classification of either biased or non-biased.

Features that were used for the classification are also output so that they
can also be analysed

@author: Jack Smith
"""
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
import pandas as pd

# initialise analysers that will be used in extracting features of all articles
vader_analyser = SentimentIntensityAnalyzer()
flair_analyser = TextClassifier.load("en-sentiment")

# regex common of all corpora being accessed
corpora_regex = "[a-zA-Z0-9-_]*.txt"

# feature word that will be counted as a feature - will be loaded in as necessary
biased_words = []
manually_extracted_biased_words = []
non_biased_words = []
subjective_words = []
objective_words = []

# feature bigrams that will be counted as a feature - will be loaded in as necessary
biased_bigrams = []
non_biased_bigrams = []
subjective_bigrams = []
objective_bigrams = []

# holds features extracted from the labelled dataset meaning it can be stored and retrieved easier in future
labeled_dataset_features = []


def load_feature(feature: list, location: str):
    """
    Used to load in a feature as a list from a pre-created text file
    :param feature: the list to populate with extracted features
    :param location: the location the features are saved
    """
    with open(location) as file:
        for line in file.readlines():
            feature.append(line.strip())


def load_all_features():
    """
    Loads all relevant features used for the :func:`extract_features` function
    """
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
    """
    Extracts all features from a piece of text that may be used to
    determine whether that text is biased or not.

    Although not copied verbatim, much of the code used in this
    function was researched at:

    Mogyorosi, M., 2021. Sentiment Analysis: First Steps With Python's NLTK Library [online].
    Available from: https://realpython.com/python-nltk-sentiment-analysis/
    [Accessed 22 February 2023].

    The code found at this reference has been built upon to extract more
    features, aiming to improve the accuracy of the supervised analysis

    :param text: the text to analyse for bias
    :return: a dict containing all features extracted from the text
    """
    # todo: try all features and then remove to see effect on accuracy

    # features of article that can be used for recognising bias
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

    for sentence in sent_tokenize(text):  # loop each sentence as VADER and Flair work best on sentences

        # loop words to check if they are in the feature words
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

        # check for bigrams in each sentence
        for bigram in biased_bigrams:
            biased_bigram_count += sentence.count(bigram)
        for bigram in non_biased_bigrams:
            non_biased_bigram_count += sentence.count(bigram)
        for bigram in subjective_bigrams:
            subjective_bigram_count += sentence.count(bigram)
        for bigram in objective_bigrams:
            objective_bigram_count += sentence.count(bigram)

        # get scores from unsupervised approaches
        compound_vader_scores.append(vader_analyser.polarity_scores(sentence)["compound"])
        textblob_scores.append(TextBlob(sentence).subjectivity)

        # get flair tag and confidence
        flair_sent = Sentence(sentence)
        flair_analyser.predict(flair_sent)
        flair_tags.append(flair_sent.tag)
        flair_confidence.append(flair_sent.score)

    # build the final output dict
    extracted_features = dict()
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


def get_labeled_features():
    if os.path.exists("/Volumes/24265241/Unsupervised Training/labeled_dataset_features.pkl"):
        with open("/Volumes/24265241/Unsupervised Training/labeled_dataset_features.pkl", "rb") as file:
            labeled_dataset_features.extend(pickle.load(file))
    else:
        biased_articles = PlaintextCorpusReader("/Volumes/24265241/Unsupervised Training/Biased", corpora_regex)
        non_biased_articles = PlaintextCorpusReader("/Volumes/24265241/Unsupervised Training/Non-biased", corpora_regex)

        labeled_dataset_features.extend([
            (extract_features(biased_articles.raw(article)), 'biased')
            for article in tqdm(biased_articles.fileids(), desc="Biased feature extraction")
        ])

        labeled_dataset_features.extend([
            (extract_features(non_biased_articles.raw(article)), 'not biased')
            for article in tqdm(non_biased_articles.fileids(), desc="Non-biased feature extraction")
        ])

        with open("/Volumes/24265241/Unsupervised Training/labeled_dataset_features.pkl", "wb") as file:
            pickle.dump(labeled_dataset_features, file)


def test_classifiers(no_tests: int):
    all_accuarcies = {
        "BernoulliNB": [],
        "ComplementNB": [],
        "MultinomialNB": [],
        "KNeighborsClassifier": [],
        "DecisionTreeClassifier": [],
        "RandomForestClassifier": [],
        "LogisticRegression": [],
        "MLPClassifier": [],
        "AdaBoostClassifier": []
    }
    for i in tqdm(range(1, no_tests + 1), desc="Models Test", colour="green"):
        train_count = (len(labeled_dataset_features) // 4) * 3  # train with 75% of the dataset

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
            "AdaBoostClassifier": AdaBoostClassifier()
        }

        for model_name, model in sklearn_classifiers.items():
            classifier = SklearnClassifier(model)
            classifier.train(labeled_dataset_features[:train_count])
            model_accuracies[model_name] = accuracy(classifier, labeled_dataset_features[train_count:])
            all_accuarcies[model_name].append(model_accuracies[model_name])

        table = tabulate(model_accuracies.items(), headers=["Model", "Accuracy"])

        with open("/Volumes/24265241/Unsupervised Training/model_accuracies.txt", "a") as file:
            file.write(f"Test {i}\n------\n")
            file.write(table + "\n\n")

    for model_name, accuracy_list in all_accuarcies.items():
        all_accuarcies[model_name] = mean(accuracy_list)

    table = tabulate(all_accuarcies.items(), headers=["Model", "Accuracy"])
    print("-- Average Accuracies --\n" + table)

    with open("/Volumes/24265241/Unsupervised Training/model_accuracies.txt", "a") as file:
        file.write("Average Scores\n--------------\n")
        file.write(table)


def majority_selection_classification(features: dict, cl1: SklearnClassifier, cl2: SklearnClassifier, cl3: SklearnClassifier):
    # make classifications from first 2 classifiers
    p1 = cl1.classify(features)
    p2 = cl2.classify(features)
    if p1 == p2:  # if classifiers 1 and 2 agree, majority found already, classifier 3 not required
        return p1
    else:  # no agreement from classifiers 1 and 2, therefore classifier 3 has the deciding vote
        return cl3.classify(features)


if __name__ == '__main__':
    # read feature word lists
    load_all_features()

    # get features for training
    get_labeled_features()

    # train various classifiers and test to find the best accuracies
    # test_classifiers(100)

    # train the best classifiers on whole dataset and store as objects
    best_classifiers = {
        "MLPClassifier": MLPClassifier(max_iter=1500),              # avg accuracy: 0.773978
        "AdaBoostClassifier": AdaBoostClassifier(),                 # avg accuracy: 0.773079
        "LogisticRegression": LogisticRegression(max_iter=1500)    # avg accuracy: 0.770218
    }

    for model_name, model in best_classifiers.items():
        classifier = SklearnClassifier(model)
        classifier.train(labeled_dataset_features)
        # save classifier obj for future use
        with open(f"/Volumes/24265241/Classifier Models/{model_name}.pkl", "wb") as f:
            pickle.dump(classifier, f)

    # load in classifiers
    with open("/Volumes/24265241/Classifier Models/MLPClassifier.pkl", "rb") as f:
        mlp_classifier: SklearnClassifier = pickle.load(f)
    with open("/Volumes/24265241/Classifier Models/AdaBoostClassifier.pkl", "rb") as f:
        ada_boost_classifier: SklearnClassifier = pickle.load(f)
    with open("/Volumes/24265241/Classifier Models/LogisticRegression.pkl", "rb") as f:
        logistic_regression_classifier: SklearnClassifier = pickle.load(f)

    # access corpora
    corpora = {
        "bbc": PlaintextCorpusReader("/Volumes/24265241/News Corpus/BBC Corpus", corpora_regex),
        "independent": PlaintextCorpusReader("/Volumes/24265241/News Corpus/Independent Corpus", corpora_regex),
        "daily_mail": PlaintextCorpusReader("/Volumes/24265241/News Corpus/Daily Mail Corpus", corpora_regex),
        "chat_gpt": PlaintextCorpusReader("/Volumes/24265241/News Corpus/ChatGPT Corpus", corpora_regex)
    }

    # analyse each corpora
    for name, corpus in tqdm(corpora.items(), desc="Corpora", colour="red", position=0, leave=False, ncols=100):
        results = []
        for fileid in tqdm(corpus.fileids(), desc=name, colour="green", position=1, leave=False, ncols=100):
            extracted_features = extract_features(corpus.raw(fileid))
            classification = majority_selection_classification(extracted_features, mlp_classifier, ada_boost_classifier, logistic_regression_classifier)

            # store file results
            results.append([
                fileid,
                classification,
                extracted_features["vader_score"],
                extracted_features["textblob_score"],
                extracted_features["flair_tag"],
                extracted_features["flair_percentage"],
                extracted_features["flair_confidence"],
            ])

        # store corpus results
        df = pd.DataFrame(results, columns=["Article", "Classification", "vader_score", "textblob_score", "flair_tag", "flair_percentage", "flair_confidence"])
        df.to_csv(f"/Volumes/24265241/Analysis Results/Supervised Results/{name}_results.csv")

