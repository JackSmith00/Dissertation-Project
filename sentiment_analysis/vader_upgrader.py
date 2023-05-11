"""
This file will extend the VADER lexicon by adding words and
their associated sentiment from 2 additional sentiment lexicons.
Words that already exist in the lexicon are not added again so that
they are not scored twice. The VADER score is taken as it has weighting
to the sentiment, whereas the new words are scored only as positive/negitive
due to a lack of further data.

The additional lexicon were found at the following resources:

Mohammad, S. M., 2011. The NRC Emotion Intensity Lexicon (NRC-EIL) [online].
Available from: http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
[Accessed 19 April 2023].

Loughran, T. & McDonald, B., 2022. Loughran-McDonald Master Dictionary w/ Sentiment Word Lists [online].
Available from: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
[Accessed 19 April 2023].
"""
from nltk.sentiment import SentimentIntensityAnalyzer

# get a copy of the original vader lexicon
vader = SentimentIntensityAnalyzer()
vader_lexicon_lines = vader.lexicon_file.split("\n")
duplicate_lines = vader_lexicon_lines.copy()  # used so items can be deleted from vader_lexicon_lines in a for-loop

# remove any emoji/symbols
for line in duplicate_lines:
    if not line[0].isalpha():
        vader_lexicon_lines.remove(line)

del duplicate_lines  # no longer needed

# -- add new entries --

# used for holding and comparing values
vader_keys = vader.lexicon.keys()  # make sure word isn't already in VADER
new_lexicon_dict = dict()  # hold new scores

# add nrc lexicon
with open("/Volumes/24265241/Additional Sentiment Lexicons/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt") as f:
    nrc_lexicon_lines = f.readlines()

for line in nrc_lexicon_lines:
    word, label, score = line.strip().split("\t")  # get line values
    if word in vader_keys or (label != "positive" and label != "negative"):
        # ignore words already in vader or with incorrect label
        continue
    if word not in new_lexicon_dict.keys():
        # if word not yet in new dict, add with appropriate value
        if score == "0":
            new_lexicon_dict[word] = 0
        elif label == "negative":
            new_lexicon_dict[word] = -2
        elif label == "positive":
            new_lexicon_dict[word] = 2
    else:  # word is already in dict, update score appropriately
        if label == "negative" and score != "0":
            new_lexicon_dict[word] -= 2
        elif label == "positive" and score != "0":
            new_lexicon_dict[word] += 2

# add Loughran-McDonald lexicon
with open("/Volumes/24265241/Additional Sentiment Lexicons/Loughran-McDonald_MasterDictionary_1993-2021.csv") as f:
    lm_lexicon_lines = f.readlines()

for line in lm_lexicon_lines:
    contents = line.strip().split(",")
    if contents[1][0].isalpha():
        # skip row of headers
        continue
    # get the relevant information from the line
    word = contents[0].lower()
    neg_score = int(contents[7])
    pos_score = int(contents[8])
    cumulative_score = pos_score - neg_score
    if word in vader_keys or word in new_lexicon_dict.keys():
        # ignore words already in vader or already scored from nrc lexicon to avoid doubling word scores
        continue
    if cumulative_score == 0:
        new_lexicon_dict[word] = 0
    elif cumulative_score > 0:
        new_lexicon_dict[word] = 2
    else:
        new_lexicon_dict[word] = -2

# incorporate new entries into vader lexicon
for word, score in new_lexicon_dict.items():
    vader_lexicon_lines.append(f"{word}\t{score}\r")

# save new vader lexicon - overwrite existing
with open("/Users/Jack/nltk_data/sentiment/vader_lexicon/vader_lexicon.txt", "w") as f:
    for line in vader_lexicon_lines:
        f.write(line + "\n")
