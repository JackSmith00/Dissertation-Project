from nltk import pos_tag, RegexpParser, ne_chunk, FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer, WordNetLemmatizer

print("--- FILTERING ---")

example_string = "Sir, I protest. I am not a merry man!"
print(f"Sentence: {example_string}")

# tokenize sentence into words
words_in_quote = word_tokenize(example_string)
print(f"Words in an array: {words_in_quote}")

# filtering stopwords
stop_words = set(stopwords.words("english"))  # retrieve pre-defined set of stop words

filtered_list = [  # list comprehension
    word for word in words_in_quote if word.casefold() not in stop_words
]

print(f"Filtered words array: {filtered_list}")

# stemming to reduce words to their root

print("\n--- STEMMING ---")

# stemmer objects to use for stemming
old_stemmer = PorterStemmer()
medium_aggression_stemmer = SnowballStemmer("english")
agressive_stemmer = LancasterStemmer()

example_string = """The crew of the USS Discovery discovered many discoveries.
Discovering is what explorers do."""
print(f"Sentence: {example_string}")

words_in_quote = word_tokenize(example_string)
print(f"Words in an array: {words_in_quote}")

# use .stem() to create an array of only word stems
old_stemmed_words = [  # list comprehension
    old_stemmer.stem(word) for word in words_in_quote
]
med_stemmed_words = [
    medium_aggression_stemmer.stem(word) for word in words_in_quote
]
agg_stemmed_words = [
    agressive_stemmer.stem(word) for word in words_in_quote
]

print(f"Old stemmed array: {old_stemmed_words}\n"
      f"Med aggression stemmed array: {med_stemmed_words}\n"
      f"Aggressive stemmed array: {agg_stemmed_words}")


print("\n--- TAGGING PARTS OF SPEECH (POS TAGGING) ---")

example_string = "If you wish to make an apple pie from scratch, you must first invent the universe."
print(f"Sentence: {example_string}")

# tokenize words
words_in_quote = word_tokenize(example_string)

# retrieve pos
words_with_pos_tags = pos_tag(words_in_quote)
print(f"Words with their POS tags: {words_with_pos_tags}\n"
      f"Meanings of tags can be printed using nltk.help.upenn_tagset()")

# now with a gibberish example
example_string = "'Twas brillig, and the slithy toves did gyre and gimble in the wabe: " \
                 "all mimsy were the borogoves, and the mome raths outgrabe."
print(f"Gibberish sentence: {example_string}")

words_in_quote = word_tokenize(example_string)
words_with_pos_tags = pos_tag(words_in_quote)
print(f"Gibberish words with their POS: {words_with_pos_tags}\n"
      f"Still half works")

print("\n--- LEMMATIZING ---")
# similar to stemming but uses real, readable words

# create a lemmatizer
lemmatizer = WordNetLemmatizer()

example_string = "The friends of DeSoto love scarves."
words_in_quote = word_tokenize(example_string)
print(f"Sentence: {example_string}")

lemmatized_words = [
    lemmatizer.lemmatize(word) for word in words_in_quote
]
print(f"Lemmatized words: {lemmatized_words}\nNOTE: plurals become singular")

print(f"Doesn't always work, for example worst becomes: {lemmatizer.lemmatize('worst')}\n"
      f"This is because 'worst' is assumed to be a noun, it can be forced to interpret as an adj using pos='a' as "
      f"a parameter\nworst -> {lemmatizer.lemmatize('worst', pos='a')}\n"
      f"Therefore, combining this method with extracting POS can help deal with homographs")

# --- CHUNKING ---

# get pos for example sentence
example_string = "It's a dangerous business, Frodo, going out your door."
print(f"Sentence: {example_string}")
words_in_quote = word_tokenize(example_string)
pos_tags = pos_tag(words_in_quote)

# define chunk grammar
grammar = "NP: {<DT>?<JJ>*<NN>}"  # NP-Noun Phrase, ?-Optional, *-Any number of

# create a chunk parser
chunk_parser = RegexpParser(grammar)

# create and draw tree of these phrases
tree = chunk_parser.parse(pos_tags)
# tree.draw()

# --- CHINKING ----

# determine what to include and exclude
grammar = """
Chunk:  {<.*>+}
        }<JJ>{"""
# includes things between {} - here everything - and exclude between }{ - here adjectives

# create chunk parser and chink
chunk_parser = RegexpParser(grammar)
tree = chunk_parser.parse(pos_tags)

print("\n--- NAMED ENTITY RECOGNITION (NER) ---")

# ne_chunk can recognise named entities
tree = ne_chunk(pos_tags)
# tree.draw()

example_string = """Men like Schiaparelli watched the red planet—it is odd, by-the-bye, that
for countless centuries Mars has been the star of war—but failed to
interpret the fluctuating appearances of the markings they mapped so well.
All that time the Martians must have been getting ready.

During the opposition of 1894 a great light was seen on the illuminated
part of the disk, first at the Lick Observatory, then by Perrotin of Nice,
and then by other observers. English readers heard of it first in the
issue of Nature dated August 2."""

print(f"Example string: {example_string}")

def extract_ne(string):
    words = word_tokenize(string)
    tags = pos_tag(words)
    tree = ne_chunk(tags, binary=True)
    return set(
        " ".join(i[0] for i in t)
        for t in tree
        if hasattr(t, "label") and t.label() == "NE"
    )

print(f"Named entities in example string: "
      f"{extract_ne(example_string)}"
      f"\n'Nice' was missed - likely because it was interpretted as an adj")

# below required nltk.book to be downloaded again (nltk.download('book'))
"""
print("\n--- CONCORDANCE ---")

text8.concordance("ship", lines=3)

print("\n--- FREQUENCY DISTRIBUTION ---")

# remove stop words
meaningful_words = [
    word for word in text8 if word.casefold() not in stop_words
]
freq_dist = FreqDist(meaningful_words)
print(freq_dist)
print(freq_dist.most_common(10))

print("\n--- COLLOCATIONS ---")
# pairs of words
print(f"Unlemmatized collocations: {text8.collocations()}")

# may be more accurate on lemmatized version
lemmatized_words = [lemmatizer.lemmatize(word) for word in text8]

# needs to be converted to an nltk text to call collocations() using nltk.Text()
lemmatized_text8 = Text(lemmatized_words)
print(f"Lemmatized collocations: {lemmatized_text8.collocations()}")
"""