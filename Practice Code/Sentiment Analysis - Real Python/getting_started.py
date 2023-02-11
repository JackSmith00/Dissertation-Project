from nltk import FreqDist, Text
from nltk.corpus import state_union, stopwords
from nltk.collocations import TrigramCollocationFinder, BigramCollocationFinder, QuadgramCollocationFinder

# --- Compiling Data ---
su_words = [word for word in state_union.words() if word.isalpha()]
# using isalpha() will prevent punctuation being included

stopwords = stopwords.words("english")
su_words_no_stops = [word for word in su_words if word.casefold() not in stopwords]  # remove any stopwords

# --- Creating Frequency Distributions ---
fd = FreqDist(su_words_no_stops)
print(fd.most_common(3))

print(f"CASE SENSITIVE: America appears {fd['America']} times, and AMERICA appears {fd['AMERICA']} times")

# Extracting Concordance and Collocations

concordance_list = Text(su_words).concordance_list("america", lines=2)
for entry in concordance_list:
    print(entry.line)

# each of the below also finds the above (trigram also finds all bigrams, etc.)
bigrams = BigramCollocationFinder.from_words(su_words)
trigrams = TrigramCollocationFinder.from_words(su_words)
quadgrams = QuadgramCollocationFinder.from_words(su_words)

print(trigrams.ngram_fd.most_common(2))
print(quadgrams.iii.most_common(2))