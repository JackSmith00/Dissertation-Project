"""
Used to test the accuracies of a range of existing classifiers trained
on features extracted from the training dataset using a StratifiedKFold
split to get an average accuracy across the whole dataset, the best of
which when tested on unseen data are used to create a `MajorityVoteClassifier`
object that is saved for future use categorising other articles.

@author: Jack Smith
"""
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

# all potential classifiers that could be used from the sklearn package
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

# load in the features extracted from the labelled dataset
with open("/Volumes/24265241/Supervised Training/labeled_dataset_features.pkl", "rb") as file:
    labelled_dataset_features = pickle.load(file)

seed(452000)  # same shuffle
shuffle(labelled_dataset_features)


def split_features(features) -> tuple[numpy, numpy]:
    """
    Splits the labelled features from its current format of a list of tuples
    containing a feature score `dict` and a label string into 2 numpy
    arrays containing the feature scores and the label
    :param features: a list of features extracted from the training dataset with their associated label
    :return: 2 numpy arrays containing the feature scores and label
    """
    X = []  # hold the features for each article in the training dataset
    y = []  # hold the true labels for each article in the training dataset
    for item in features:
        X.append(list(item[0].values()))  # add only the dict values as they are all that is needed
        y.append(item[1])  # add the associated label
    return numpy.array(X), numpy.array(y)  # return the extracted lists as numpy arrays


# features are stored in a dict not numpy arrays currently - needs splitting
X, y = split_features(labelled_dataset_features)

# separate the 1st 10% of the dataset to use as testing data
strat_k_fold = StratifiedKFold(n_splits=10, shuffle=True)
(train_indices, test_indices), *fold = strat_k_fold.split(X, y)

X_test = X[test_indices]
X_10_fold = X[train_indices]
y_test = y[test_indices]
y_10_fold = y[train_indices]

# test all classifiers of which best 3 will be used for majority vote

accuracies = {  # will hold all recorded accuracies for each classifier
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
    for classifier_name, classifier in all_classifiers.items():  # loop each classifier
        classifier.fit(X_10_fold[train_index], y_10_fold[train_index])  # train classifier
        accuracy = classifier.score(X_10_fold[validation_index], y_10_fold[validation_index])  # find accuracy
        accuracies[classifier_name].append((classifier, accuracy))  # store model and accuracy

# will be used to store the best version of each classifier in the form (classifier, accuracy)
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
        }  # current values are placeholders that will be replaced when comparison begins

# find the most accurate version of each classifier
for classifier_name, k_fold in accuracies.items():
    for classifier, accuracy in k_fold:  # get the model and accuracy for each k-fold test
        if accuracy > best_of_classifiers[classifier_name][1]:  # if better than currently held, replace
            best_of_classifiers[classifier_name] = (classifier, accuracy)

# test the best versions of each model on unseen data previously separated to get true accuracy scores

true_accuracies = []  # to hold the accuracies of the best classifiers tested on the unseen data

for (classifier, accuracy) in best_of_classifiers.values():
    # test each classifier on unseen data and record the accuracy
    true_accuracies.append((classifier, classifier.score(X_test, y_test)))

# order classifiers based on accuracy
"""
The code to sort the accuracies based on the second value of a tuple rather
than the first was found at the following resource:

Gallagher, J., 2020. How to Sort a Dictionary by Value in Python. [Online] 
Available at: https://careerkarma.com/blog/python-sort-a-dictionary-by-value/
[Accessed 1 May 2023]
"""
sorted_accuracies = sorted(true_accuracies, key=lambda x: x[1], reverse=True)

best_3_classifiers = sorted_accuracies[:3]  # seperate best 3 accuracies to use in majority vote classifier

# initialise and test majority selection
mvc = MajorityVoteClassifier(best_3_classifiers[0][0], best_3_classifiers[1][0], best_3_classifiers[2][0])

correct = 0  # to test MajorityVoteClassifier, count how many test articles it correctly labels

for i in range(len(X_test)):  # loop each test article
    if mvc.predict([X_test[i]]) == y_test[i]:
        correct += 1

mvc_accuracy = correct / len(X_test)  # accuracy is total correct predictions / total predictions made

print(f"Majority Vote Accuracy = {mvc_accuracy}")

# save majority vote classifier with accuracy indication
with open(f"/Volumes/24265241/Majority Vote Classifier/majority_vote_classifier{round(mvc_accuracy * 100)}.pkl", "wb") as f:
    pickle.dump(mvc, f)
