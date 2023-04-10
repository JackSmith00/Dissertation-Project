import nltk.corpus
from nltk.corpus import PlaintextCorpusReader
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from statistics import mean, mode
from textblob import TextBlob
from flair.models import TextClassifier
from flair.data import Sentence
import pandas as pd
from tqdm import tqdm

corpora_regex = "[a-zA-Z0-9-]*.txt"


def analyse_corpus(corpus: nltk.corpus.CorpusReader) -> pd.DataFrame:
    # initialise the VADER and Flair analysers
    vader = SentimentIntensityAnalyzer()
    flair = TextClassifier.load("en-sentiment")

    # get file ids
    files = corpus.fileids()

    data = []  # hold all retrieved data
    for file in tqdm(files, desc="Corpus"):  # loop every file for analysis
        sentences = sent_tokenize(corpus.raw(file))  # tokenise sentences
        # create lists to hold scores for each sentence in the article
        pos_scores = []
        neg_scores = []
        comp_scores = []
        blob_scores = []
        flair_tags = []
        flair_confidence = []

        for sentence in sentences:  # loop each sentence
            # get scores from VADER and Flair analysis
            scores = vader.polarity_scores(sentence)
            flair_sent = Sentence(sentence)
            flair.predict(flair_sent)

            # append retrieved scores to the corresponding list for the file
            pos_scores.append(scores["pos"])
            neg_scores.append(scores["neg"])
            comp_scores.append(scores["compound"])
            blob_scores.append(TextBlob(sentence).subjectivity)
            flair_tags.append(flair_sent.tag)
            flair_confidence.append(flair_sent.score)

        # append the averages of each attribute to the overall data list for the corpus
        data.append([mean(pos_scores),
                     mean(neg_scores),
                     mean(comp_scores),
                     mean(blob_scores),
                     mode(flair_tags),
                     flair_tags.count(mode(flair_tags)) / len(flair_tags),  # percentage of articles with the modal tag
                     mean(flair_confidence)])

    # create a dataframe to hold all info
    return pd.DataFrame(data, columns=["Pos", "Neg", "Comp", "Subj", "Flair Label", "Label Percentage", "Flair Conf"])


if __name__ == '__main__':
    # Step 1 - Read the Corpora
    # bbc_corpus = PlaintextCorpusReader("/Volumes/24265241/News Corpus/BBC Corpus", corpora_regex)
    # independent_corpus = PlaintextCorpusReader("/Volumes/24265241/News Corpus/Independent Corpus", corpora_regex)
    daily_mail_corpus = PlaintextCorpusReader("/Volumes/24265241/News Corpus/Daily Mail Corpus", corpora_regex)
    chat_gpt_corpus = PlaintextCorpusReader("/Volumes/24265241/News Corpus/ChatGPT Corpus", corpora_regex)

    # Step 2 - Perform analysis
    # bbc_df = analyse_corpus(bbc_corpus)
    # independent_df = analyse_corpus(independent_corpus)
    daily_mail_df = analyse_corpus(daily_mail_corpus)
    daily_mail_df.to_csv("/Volumes/24265241/Analysis Results/daily_mail_df.csv")  # move back
    chat_gpt_df = analyse_corpus(chat_gpt_corpus)

    # Step 3 - Export analysis data to CSV
    # bbc_df.to_csv("/Volumes/24265241/Analysis Results/bbc_df.csv")
    # independent_df.to_csv("/Volumes/24265241/Analysis Results/independent_df.csv")
    chat_gpt_df.to_csv("/Volumes/24265241/Analysis Results/chat_gpt_df.csv")
