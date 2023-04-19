from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
import pickle
from random import shuffle, seed
import numpy
from tqdm import tqdm
from MajorityVoteClassifier import MajorityVoteClassifier

all_classifiers = {
            "BernoulliNB": BernoulliNB(),
            "ComplementNB": ComplementNB(),
            "MultinomialNB": MultinomialNB(),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "RandomForestClassifier": RandomForestClassifier(),
            "LogisticRegression": LogisticRegression(max_iter=1500),
            "MLPClassifier": MLPClassifier(max_iter=1500),
            "AdaBoostClassifier": AdaBoostClassifier()
        }

with open("/Volumes/24265241/Supervised Training/labeled_dataset_features.pkl", "rb") as file:
    labeled_dataset_features = pickle.load(file)

seed(452000)  # same shuffle
shuffle(labeled_dataset_features)


def split_features(features) -> tuple[numpy, numpy]:
    X = []
    y = []
    for item in features:
        X.append(list(item[0].values()))
        y.append(item[1])
    return numpy.array(X), numpy.array(y)


# features are stored in a dict not numpy arrays currently - needs splitting
X, y = split_features(labeled_dataset_features)

# separate the 1st 10% of the dataset to use as testing data
strat_k_fold = StratifiedKFold(n_splits=10, shuffle=True)
(train_indices, test_indices), *fold = strat_k_fold.split(X, y)

X_10_fold = X[train_indices]
X_test = X[test_indices]
y_10_fold = y[train_indices]
y_test = y[test_indices]

# test all classifiers of which best 3 will be used for majority vote

# will hold all recorded accuracies for each classifier
accuracies = {
            "BernoulliNB": [],
            "ComplementNB": [],
            "MultinomialNB": [],
            "KNeighborsClassifier": [],
            "DecisionTreeClassifier": [],
            "RandomForestClassifier": [],
            "LogisticRegression": [],
            "MLPClassifier": [],
            "AdaBoostClassifier": []
        }

# perform Strat K Fold training and validation on each of the classifiers
for train_index, validation_index in tqdm(strat_k_fold.split(X_10_fold, y_10_fold)):
    for classifier_name, classifier in all_classifiers.items():
        classifier.fit(X_10_fold[train_index], y_10_fold[train_index])
        accuracy = classifier.score(X_10_fold[validation_index], y_10_fold[validation_index])
        accuracies[classifier_name].append((classifier, accuracy))

# will be used to store the best version of each classifier
best_of_classifiers = {
            "BernoulliNB": (None, 0),
            "ComplementNB": (None, 0),
            "MultinomialNB": (None, 0),
            "KNeighborsClassifier": (None, 0),
            "DecisionTreeClassifier": (None, 0),
            "RandomForestClassifier": (None, 0),
            "LogisticRegression": (None, 0),
            "MLPClassifier": (None, 0),
            "AdaBoostClassifier": (None, 0)
        }

# find the most accurate version of each classifier
for classifier_name, k_fold in accuracies.items():
    for classifier, accuracy in k_fold:
        if accuracy > best_of_classifiers[classifier_name][1]:
            best_of_classifiers[classifier_name] = (classifier, accuracy)

# to hold the accuracies of the best classifiers tested on the unseen data
true_accuracies = []

for (classifier, accuracy) in best_of_classifiers.values():
    true_accuracies.append((classifier, classifier.score(X_test, y_test)))

sorted_accuracies = sorted(true_accuracies, key=lambda x: x[1], reverse=True)
# https://careerkarma.com/blog/python-sort-a-dictionary-by-value/
# reference this for how to sort tuple based on 2nd value


best_3_classifiers = sorted_accuracies[:3]


# test majority selection
mvc = MajorityVoteClassifier(best_3_classifiers[0][0], best_3_classifiers[1][0], best_3_classifiers[2][0])

correct = 0

for i in range(len(X_test)):
    if mvc.predict([X_test[i]]) == y_test[i]:
        correct += 1

mvc_accuracy = correct / len(X_test)

print(f"Majority Vote Accuracy = {mvc_accuracy}")

# save majority vote classifier
with open(f"/Volumes/24265241/Majority Vote Classifier/majority_vote_classifier{round(mvc_accuracy * 100)}.pkl", "wb") as f:
    pickle.dump(mvc, f)
