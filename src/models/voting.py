import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets


def soft_transform(clfs, embeds):
    probas = np.array([clf.predict_proba(embeds[i]) for i, clf in enumerate(clfs)])
    return np.argmax(np.average(probas, axis=0), axis=1)


def hard_transform(clfs, embeds):
    preds = np.array([clf.predict(embeds[i]) for i, clf in enumerate(clfs)])
    return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), arr=preds, axis=1)


class VotingClassifierPretrained(VotingClassifier):
    def __init__(self, estimators, voting='hard', weights=None, n_jobs=None,
                 flatten_transform=True, pretrained=True):
        super().__init__(estimators, voting, weights, n_jobs, flatten_transform)
        self.pretrained = pretrained

    def _fit_pretrained(self, y):
        check_classification_targets(y)
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError("Multilabel and multi-output classification is not supported.")

        if self.voting not in ('soft', 'hard'):
            raise ValueError("Voting must be 'soft' or 'hard'; got {!r}".format(self.voting))

        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        return self

    def fit(self, X, y, sample_weight=None):
        if self.pretrained:
            return self._fit_pretrained(y)
        return super().fit(X, y, sample_weight)
