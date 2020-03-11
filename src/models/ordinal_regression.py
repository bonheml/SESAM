from sklearn.base import clone, ClassifierMixin, MetaEstimatorMixin, BaseEstimator
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, check_array


class OrdinalClassifier(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    """ This ordinal classifier uses the approach proposed by [1].
    It should not be used when the number of ordinal labels is large as the results are more noisy.
    The implementation is a modified version of the one proposed by [2].

    [1] Frank, E. and Hall, M., 2001, September. A simple approach to ordinal classification.
    In European Conference on Machine Learning (pp. 145-156). Springer, Berlin, Heidelberg.
    [2] https://towardsdatascience.com/simple-trick-to-train-an-ordinal-regression-with-any-classifier-6911183d2a3c
    """
    def __init__(self, estimator=LogisticRegression()):
        """Initialise the wrapper with the classifier to use and the labels

        :param estimator: Which classifier to use. Can be any sklearn binary classifier
        """
        self.estimator = estimator

    def _more_tags(self):
        return {'multioutput_only': True}

    def _fit_multi(self):
        classes_list = list(self.classes_)
        for i in range(self.classes_.shape[0] - 1):
            clf = clone(self.estimator)
            binary_y = np.array(list(map(lambda x: classes_list.index(x) > i, self.y_)))
            clf.fit(self.X_, binary_y)
            self.estimators_[self.classes_[i]] = clf

    def fit(self, X, y):
        """ Fit the n_labels -1 classifiers

        :param X  array-like, shape (n_samples, n_features) The training input samples.
        :param y: array-like, shape (n_samples,) The target values. An array of int.
        :return: Returns self.
        :rtype: object
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.y_ = y
        self.X_ = X

        self.classes_, y = np.unique(y, return_inverse=True)
        self.estimators_ = {}

        if len(self.classes_) == 1:
            self.estimator.fit(self.X_, self.y_)
        else:
            self._fit_multi()

        return self

    def _predict_multi(self, X):
        clfs_pred = {i: clf.predict_proba(X) for i, clf in self.estimators_.items()}
        n_classes = self.classes_.shape[0] - 1
        predicted_proba = []

        for i, name in enumerate(self.classes_):
            prev_pred = 1 if i == 0 else clfs_pred[self.classes_[i - 1]][:, 1]
            curr_pred = 0 if i >= n_classes else clfs_pred[self.classes_[i]][:, 1]
            predicted_proba.append(prev_pred - curr_pred)

        return predicted_proba

    def predict_proba(self, X):
        """Predict the probability of each ordinal label

        :param X: input features
        :return: list of predicted probabilities
        """
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)

        if len(self.classes_) == 1:
            predicted_proba = self.estimator.predict_proba(X)
        else:
            predicted_proba = self._predict_multi(X)

        return np.vstack(predicted_proba).T

    def predict(self, X):
        """Predict the label of X

        :param X: input features
        :return: the predicted label
        """
        check_is_fitted(self, ['X_', 'y_'])
        if len(self.classes_) == 1:
            return self.estimator.predict(X)
        D = self.predict_proba(X)
        return self.classes_[np.argmax(D, axis=1)]
