"""
Used to extract words from the labelled dataset to use as features for
recognising biased news articles. The labelled dataset used is referenced
below:

Spinde, T., 2021. MBIC – A Media Bias Annotation Dataset [online].
Available from: https://www.kaggle.com/datasets/timospinde/mbic-a-media-bias-annotation-dataset?resource=download
[Accessed 19 April 2023].

The code in this file has been researched at the resource below. It has
been adapted to suit the needs of this project.

Mogyorosi, M., 2021. Sentiment Analysis: First Steps With Python's NLTK Library [online].
Available from: https://realpython.com/python-nltk-sentiment-analysis/
[Accessed 22 February 2023].
"""
from nltk.corpus import subjectivity, stopwords, names, PlaintextCorpusReader
from nltk.collocations import BigramCollocationFinder
from nltk import pos_tag, FreqDist

# create a list of irrelevant words (stop words and names)
words_to_ignore = stopwords.words("english")
words_to_ignore.extend([name.lower() for name in names.words()])


def skip_irrelevant_words(pos_tuple: set) -> bool:
    """
    Called to determine if a word should be ignored when
    building feature word set. Ignores any word in `words_to_ignore`
    (stopwords and names) as well as words with numerics or symbols
    and any other nouns that would not indicate sentiment.

    :param pos_tuple: A pos tuple of the word to check relevance for
    :return: `True` when word should be included in feature words,
            `False` when word should be skipped
    """
    word, tag = pos_tuple
    if word.casefold() in words_to_ignore or not word.isalpha():
        return False
    if tag.startswith("NN"):
        return False
    return True


def extract_feature_words(corpus_words: list[str]) -> FreqDist:
    """
    Takes all words from the corpus, removes any that are not
    relevant and then returns them as a `FreqDist`
    :param corpus_words: A list of all words in the corpus
    :return: `FreqDist` showing the frequency of all relevant words in the corpus
    """
    return FreqDist([word.lower() for word, tag in filter(
        skip_irrelevant_words,
        pos_tag(corpus_words)
    )])


def extract_feature_bigrams(corpus_words: list[str]) -> FreqDist:
    """
    Takes all words from the corpus, removes any that are not
    relevant and then returns them as a `FreqDist` in pairs (bigrams)
    :param corpus_words: A list of all words in the corpus
    :return: `FreqDist` showing the frequency of all relevant bigrams in the corpus
    """
    return BigramCollocationFinder.from_words([
        word.lower() for word in corpus_words
        if word.casefold() not in words_to_ignore and word.isalpha()
    ]).ngram_fd


def remove_common_features(fd1: FreqDist, fd2: FreqDist):
    """
    Removes any items that are common to both `FreqDist`
    as these may not be a good indicator for either.
    E.g. Words that appear in both biased and unbiased
    articles don't necesarrily indicate the presence of bias
    :param fd1: the first `FreqDist` to remove duplicates from
    :param fd2: the second `FreqDist` to remove duplicates from
    """
    common_set = set(fd1).intersection(fd2)
    for item in common_set:
        del fd1[item]
        del fd2[item]


def remove_common_features_3_sets(fd1: FreqDist, fd2: FreqDist, fd3: FreqDist):
    """
    Removes any items that are common to all 3 `FreqDist`
    as well as common between each of the `FreqDist`
    :param fd1: the first `FreqDist` to remove duplicates from
    :param fd2: the second `FreqDist` to remove duplicates from
    :param fd3: the third `FreqDist` to remove duplicates from
    """
    # remove the features common to all fd
    common_set = set(fd1).intersection(fd2, fd3)
    for item in common_set:
        del fd1[item]
        del fd2[item]
        del fd3[item]

    # remove the features common between each combination of fd
    remove_common_features(fd1, fd2)
    remove_common_features(fd1, fd3)
    remove_common_features(fd2, fd3)


def save_features(features: list[str, int] | list[tuple, int], save_location: str):
    """
    Saves the features to an external file so that they can easily be accessed
    in the future, without the need to extract them again
    :param features: the features to save from a FreqDist (only the feature is saved, not the frequency)
    :param save_location: path to the file save location
    """
    with open(save_location, "w") as f:
        for item in features:
            if item[0].__class__ == tuple:  # to handle saving bigrams, saving the words on one line
                f.write(" ".join(item[0]) + "\n")  # access only the words
            else:  # handling individual words
                f.write(item[0] + "\n")  # save only the word


if __name__ == '__main__':

    # extract features from subjectivity corpus
    subjective_words = extract_feature_words(subjectivity.words(categories="subj"))
    subjective_bigrams = extract_feature_bigrams(subjectivity.words(categories="subj"))

    objective_words = extract_feature_words(subjectivity.words(categories="obj"))
    objective_bigrams = extract_feature_bigrams(subjectivity.words(categories="obj"))

    # remove common features
    remove_common_features(subjective_words, objective_words)
    remove_common_features(subjective_bigrams, objective_bigrams)

    # save subjectivity features
    save_features(subjective_words.most_common(200), "features/subjective_words.txt")
    save_features(objective_words.most_common(200), "features/objective_words.txt")
    save_features(subjective_bigrams.most_common(30), "features/subjective_bigrams.txt")
    save_features(objective_bigrams.most_common(30), "features/objective_bigrams.txt")

    # extract features from media bias corpus
    corpora_regex = "[a-zA-Z0-9-_]*.txt"

    biased_corpus = PlaintextCorpusReader("/Volumes/24265241/Supervised Training/Biased", corpora_regex)
    non_biased_corpus = PlaintextCorpusReader("/Volumes/24265241/Supervised Training/Non-biased", corpora_regex)
    inconclusive_bias_corpus = PlaintextCorpusReader("/Volumes/24265241/Supervised Training/Unclassified", corpora_regex)

    biased_words = extract_feature_words(biased_corpus.words())
    biased_bigrams = extract_feature_bigrams(biased_corpus.words())

    non_biased_words = extract_feature_words(non_biased_corpus.words())
    non_biased_bigrams = extract_feature_bigrams(non_biased_corpus.words())

    inconclusive_words = extract_feature_words(inconclusive_bias_corpus.words())
    inconclusive_bigrams = extract_feature_bigrams(inconclusive_bias_corpus.words())

    # remove common features
    remove_common_features_3_sets(biased_words, non_biased_words, inconclusive_words)
    remove_common_features_3_sets(biased_bigrams, non_biased_bigrams, inconclusive_bigrams)

    # save media bias features
    save_features(biased_words.most_common(200), "features/biased_words.txt")
    save_features(non_biased_words.most_common(200), "features/non_biased_words.txt")
    save_features(biased_bigrams.most_common(30), "features/biased_bigrams.txt")
    save_features(non_biased_bigrams.most_common(30), "features/non_biased_bigrams.txt")
    save_features(inconclusive_words.most_common(200), "features/inconclusive_words.txt")
    save_features(inconclusive_bigrams.most_common(30), "features/inconclusive_bigrams.txt")
