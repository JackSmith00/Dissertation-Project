class MajorityVoteClassifier:

    def __init__(self, classifier1, classifier2, classifier3):
        self.classifier1 = classifier1
        self.classifier2 = classifier2
        self.classifier3 = classifier3

    def predict(self, X):
        prediction1 = self.classifier1.predict(X)
        prediction2 = self.classifier2.predict(X)
        if prediction1 == prediction2:  # if classifiers 1 and 2 agree, majority found already, classifier 3 not required
            return prediction1
        else:  # no agreement from classifiers 1 and 2, therefore classifier 3 has the deciding vote
            return self.classifier3.predict(X)
