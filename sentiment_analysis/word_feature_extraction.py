from nltk.corpus import subjectivity, stopwords, names, PlaintextCorpusReader, LazyCorpusLoader
from nltk.collocations import BigramCollocationFinder
from nltk import pos_tag, FreqDist

words_to_ignore = stopwords.words("english")
words_to_ignore.extend([name.lower() for name in names.words()])


def skip_irrelevant_words(pos_tuple: set):
    # todo: ref sent analysis real python
    word, tag = pos_tuple
    if word.casefold() in words_to_ignore or not word.isalpha():
        return False
    if tag.startswith("NN"):
        return False
    return True


def extract_feature_words(corpus_words: list[str]) -> FreqDist:
    # todo: ref sent analysis real python
    return FreqDist([word.lower() for word, tag in filter(
        skip_irrelevant_words,
        pos_tag(corpus_words)
    )])


def extract_feature_bigrams(corpus_words: list[str]) -> FreqDist:
    # todo: ref sent analysis real python
    return BigramCollocationFinder.from_words([
        word.lower() for word in corpus_words
        if word.casefold() not in words_to_ignore and word.isalpha()
    ]).ngram_fd


def remove_common_features(fd1: FreqDist, fd2: FreqDist):
    common_set = set(fd1).intersection(fd2)
    for item in common_set:
        del fd1[item]
        del fd2[item]


def save_features(features: list[str, int] | list[tuple, int], save_location: str):
    with open(save_location, "w") as f:
        for item in features:
            if item[0].__class__ == tuple:
                f.write(" ".join(item[0]) + "\n")
            else:
                f.write(item[0] + "\n")


if __name__ == '__main__':

    # extract features from subjectivity corpus
    subjective_words = extract_feature_words(subjectivity.words(categories="subj"))
    subjective_bigrams = extract_feature_bigrams(subjectivity.words(categories="subj"))

    objective_words = extract_feature_words(subjectivity.words(categories="obj"))
    objective_bigrams = extract_feature_bigrams(subjectivity.words(categories="obj"))

    remove_common_features(subjective_words, objective_words)
    remove_common_features(subjective_bigrams, objective_bigrams)

    # # # save subjectivity features
    save_features(subjective_words.most_common(150), "features/subjective_words.txt")
    save_features(objective_words.most_common(150), "features/objective_words.txt")
    save_features(subjective_bigrams.most_common(30), "features/subjective_bigrams.txt")
    save_features(objective_bigrams.most_common(30), "features/objective_bigrams.txt")

    # extract features from media bias corpus
    corpora_regex = "[a-zA-Z0-9-_]*.txt"

    biased_corpus = PlaintextCorpusReader("/Volumes/24265241/Unsupervised Training/Biased", corpora_regex)
    non_biased_corpus = PlaintextCorpusReader("/Volumes/24265241/Unsupervised Training/Non-biased", corpora_regex)

    biased_words = extract_feature_words(biased_corpus.words())
    biased_bigrams = extract_feature_bigrams(biased_corpus.words())

    non_biased_words = extract_feature_words(non_biased_corpus.words())
    non_biased_bigrams = extract_feature_bigrams(non_biased_corpus.words())

    remove_common_features(biased_words, non_biased_words)
    remove_common_features(biased_bigrams, non_biased_bigrams)

    # save media bias features
    save_features(biased_words.most_common(150), "features/biased_words.txt")
    save_features(non_biased_words.most_common(150), "features/non_biased_words.txt")
    save_features(biased_bigrams.most_common(30), "features/biased_bigrams.txt")
    save_features(non_biased_bigrams.most_common(30), "features/non_biased_bigrams.txt")
