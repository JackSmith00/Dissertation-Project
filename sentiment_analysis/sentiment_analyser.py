from nltk.corpus import PlaintextCorpusReader
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from statistics import mean, mode
from textblob import TextBlob
from flair.models import TextClassifier
from flair.data import Sentence
import pandas as pd

# Step 1 - Read the Corpus
bbc_corpus = PlaintextCorpusReader("/Volumes/24265241/BBC Corpus/", "[a-zA-Z0-9-]*.txt")
files = bbc_corpus.fileids()

vader = SentimentIntensityAnalyzer()

flair = TextClassifier.load("en-sentiment")

data = []

i = 0

for file in files:
    sentences = sent_tokenize(bbc_corpus.raw(file))
    pos_scores = []
    neg_scores = []
    comp_scores = []
    blob_scores = []
    flair_tags = []
    flair_confidence = []

    for sentence in sentences:
        scores = vader.polarity_scores(sentence)
        flair_sent = Sentence(sentence)
        flair.predict(flair_sent)

        pos_scores.append(scores["pos"])
        neg_scores.append(scores["neg"])
        comp_scores.append(scores["compound"])
        blob_scores.append(TextBlob(sentence).subjectivity)
        flair_tags.append(flair_sent.tag)
        flair_confidence.append(flair_sent.score)

    data.append([mean(pos_scores),
                 mean(neg_scores),
                 mean(comp_scores),
                 mean(blob_scores),
                 mode(flair_tags),
                 flair_tags.count(mode(flair_tags)) / len(flair_tags),
                 mean(flair_confidence)])

df = pd.DataFrame(data, columns=["Pos", "Neg", "Comp", "Subj", "Flair Label", "Label Percentage", "Flair Conf"])

print(df.head())
print(df[["Pos", "Neg", "Comp", "Subj"]].mean())
print(df[["Flair Label"]].mode())
