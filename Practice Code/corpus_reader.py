from nltk.corpus import PlaintextCorpusReader

corpus_reader = PlaintextCorpusReader("/Volumes/24265241/BBC Corpus/", "[a-zA-Z0-9-]*.txt")

print(corpus_reader.fileids())