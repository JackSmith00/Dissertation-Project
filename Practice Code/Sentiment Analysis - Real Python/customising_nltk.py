import nltk.corpus
from nltk.sentiment import SentimentIntensityAnalyzer
from statistics import mean
from random import shuffle
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

unwanted = nltk.corpus.stopwords.words("english")
unwanted.extend([w.lower() for w in nltk.corpus.names.words()])  # ignore any names that may appear


def skip_unwanted(pos_tuple: set) -> bool:
    word, tag = pos_tuple
    if not word.isalpha() or word in unwanted:
        return False
    if tag.startswith("NN"):  # ignore nouns
        return False
    return True


# * - not needed anymore as now stored in a file for speed

# *
# positive_words = [word for word, tag in filter(  # all relevant words in positive reviews
#     skip_unwanted,
#     nltk.pos_tag(nltk.corpus.movie_reviews.words(categories=["pos"]))
# )]
# negative_words = [word for word, tag in filter(  # all relevant words in negative reviews
#     skip_unwanted,
#     nltk.pos_tag(nltk.corpus.movie_reviews.words(categories=["neg"]))
# )]
# *
# positive_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words([
#     w for w in nltk.corpus.movie_reviews.words(categories=["pos"])
#     if w.isalpha() and w not in unwanted
# ])
# negative_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words([
#     w for w in nltk.corpus.movie_reviews.words(categories=["neg"])
#     if w.isalpha() and w not in unwanted
# ])

# *
# create frequency distributions of all words in these reviews
# pos_fd = nltk.FreqDist(positive_words)
# neg_fd = nltk.FreqDist(negative_words)

# remove words that common to both types of review
# common_set = set(pos_fd).intersection(neg_fd) *
# common_bigrams = set(positive_bigram_finder.ngram_fd).intersection(negative_bigram_finder.ngram_fd)
# for word in common_set: *
#     del pos_fd[word]
#     del neg_fd[word]

# for bigram in common_bigrams:
#     del positive_bigram_finder.ngram_fd[bigram]
#     del negative_bigram_finder.ngram_fd[bigram]

with open("features/top_100_positive_words") as file:
    top_100_positive_words = file.read().split("\n")

with open("features/top_100_negative_words") as file:
    top_100_negative_words = file.read().split("\n")

with open("features/top_50_positive_bigrams") as file:
    top_50_positive_bigrams = file.read().split("\n")

with open("features/top_50_negative_bigrams") as file:
    top_50_negative_bigrams = file.read().split("\n")

print(f"Top 10 positive words are: " + ", ".join(top_100_positive_words[:10]))
print(f"Top 10 negative words are: " + ", ".join(top_100_negative_words[:10]))
print(f"Top 10 positive bigrams are: " + ", ".join(top_50_positive_bigrams[:10]))
print(f"Top 10 negative bigrams are: " + ", ".join(top_50_negative_bigrams[:10]))


# extracting features for each piece of text
def extract_features(text) -> dict:
    features = dict()
    wordcount = 0
    compound_scores = list()
    positive_scores = list()

    sia = SentimentIntensityAnalyzer()

    for sentence in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sentence):
            if word.casefold() in top_100_positive_words:
                wordcount += 1
        compound_scores.append(sia.polarity_scores(sentence)["compound"])
        positive_scores.append(sia.polarity_scores(sentence)["pos"])

    # Adding 1 to the final compound score to always have positive numbers
    # since some classifiers you'll use later don't work with negative numbers.
    features["mean_compound"] = mean(compound_scores) + 1
    features["mean_positive"] = mean(positive_scores)
    features["wordcount"] = wordcount

    return features


# hold these features in arrays of tuples, with pointers to whether they were positive or negative
features = [
    (extract_features(nltk.corpus.movie_reviews.raw(review)), 'pos')
    for review in nltk.corpus.movie_reviews.fileids(categories="pos")
]
features.extend([
    (extract_features(nltk.corpus.movie_reviews.raw(review)), "neg")
    for review in nltk.corpus.movie_reviews.fileids(categories="neg")
])


# train the classifier

train_count = len(features) // 4  # use 1/4 of the dataset for training purposes
shuffle(features)  # avoid accidentally grouping similarly classified reviews in the first quarter of the list

naive_bayes_classifier = nltk.NaiveBayesClassifier.train(features[:train_count])
naive_bayes_classifier.show_most_informative_features(10)
print(f"{nltk.classify.accuracy(naive_bayes_classifier, features[train_count:]):.2%} - NaiveBayesClassifier")

sklearn_classifiers = {
    "BernoulliNB": BernoulliNB(),
    "ComplementNB": ComplementNB(),
    "MultinomialNB": MultinomialNB(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(),
    "MLPClassifier": MLPClassifier(max_iter=1000),
    "AdaBoostClassifier": AdaBoostClassifier(),
}
for name, sklearn_classifier in sklearn_classifiers.items():
    classifier = nltk.classify.SklearnClassifier(sklearn_classifier)
    classifier.train(features[:train_count])
    accuracy = nltk.classify.accuracy(classifier, features[train_count:])
    print(f"{accuracy:.2%} - {name}")
