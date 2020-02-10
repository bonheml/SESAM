from sklearn.base import clone
import numpy as np


class OrdinalClassifier:
    """ This ordinal classifier uses the approach proposed by [1].
    It should not be used when the number of ordinal labels is large as the results are more noisy.
    The implementation is a modified version of the one proposed by [2].

    [1] Frank, E. and Hall, M., 2001, September. A simple approach to ordinal classification.
    In European Conference on Machine Learning (pp. 145-156). Springer, Berlin, Heidelberg.
    [2] https://towardsdatascience.com/simple-trick-to-train-an-ordinal-regression-with-any-classifier-6911183d2a3c
    """
    def __init__(self, clf, labels):
        """Initialise the wrapper with the classifier to use and the labels

        :param clf: Which classifier to use. Can be any sklearn binary classifier
        :param labels: labels used for the classification
        """
        self.clfs = {i: clone(clf) for i in range(len(labels) - 1)}
        self.labels = labels

    def _check_y_labels(self, y):
        y_labels = np.unique(y)
        try:
            assert set(y_labels) == set(self.labels)
        except AssertionError:
            raise AssertionError("The labels {} are not matching the given classes {}".format(y_labels, self.labels))

    def fit(self, X, y):
        self._check_y_labels(y)
        for k, clf in self.clfs.items():
            binary_y = self.labels.index(y) > k
            clf.fit(X, binary_y)

    def predict_proba(self, X):
        clfs_pred = {i: clf.predict_proba(X) for i, clf in self.clfs.items()}
        n_classes = self.labels.shape[0] - 1
        predicted_proba = []

        for i, name in self.labels:
            prev_pred = 1 if i == 0 else clfs_pred[i - 1]
            curr_pred = 0 if i >= n_classes else clfs_pred[i]
            predicted_proba.append(prev_pred - curr_pred)

        return predicted_proba

    def predict(self, X):
        pred_label_idx = np.argmax(self.predict_proba(X))
        return pred_label_idx, self.labels[pred_label_idx]
