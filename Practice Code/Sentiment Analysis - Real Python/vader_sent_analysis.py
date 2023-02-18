from nltk import sent_tokenize
from nltk.corpus import twitter_samples, movie_reviews
from nltk.sentiment import SentimentIntensityAnalyzer
from random import shuffle
from statistics import mean

# --- Sentiment Analysis ---

# create analyser instance
sia = SentimentIntensityAnalyzer()

# get polarity scores of some raw text
print(sia.polarity_scores("Wow, NLTK is really powerful!"))

# get the twitter corpus as an array of strings
tweets = [t.replace("://", "//") for t in twitter_samples.strings()]  # removing :// prevents any links working


def is_positive(tweet: str) -> bool:
    """True if a tweet has a positive compound sentiment, False otherwise"""
    return sia.polarity_scores(tweet)["compound"] > 0


shuffle(tweets)  # randomly order the tweets
for tweet in tweets[:3]:
    print(">", is_positive(tweet), tweet)

# now on movie reviews
positive_review_ids = movie_reviews.fileids(categories=["pos"])
negative_review_ids = movie_reviews.fileids(categories=["neg"])
all_review_ids = positive_review_ids + negative_review_ids


def is_positive_movie(review_id: str) -> bool:
    """True if the average of all sentence compound scores is positive."""
    text = movie_reviews.raw(review_id)
    scores = [
        sia.polarity_scores(sentence)["compound"]
        for sentence in sent_tokenize(text)
    ]
    return mean(scores) > 0


shuffle(all_review_ids)
correct = 0
for review_id in all_review_ids:
    if is_positive_movie(review_id):
        if review_id in positive_review_ids:
            correct += 1
    else:
        if review_id in negative_review_ids:
            correct += 1

print(F"{correct / len(all_review_ids):.2%} correct")
