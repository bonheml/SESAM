import numpy as np


def soft_transform(clfs, embeds, multilabel, multitask):
    probas = [clf.predict_proba(embeds[i]) for i, clf in enumerate(clfs)]
    probas = list(zip(*probas))
    if multitask:
        res = []
        for p in probas:
            p = np.rollaxis(np.array(p), 1, 0)
            res.append(np.argmax(np.average(p, axis=1), axis=1))
        return np.array(list(zip(*res)))

    probas_avg = np.average(np.array(probas), axis=1)
    if multilabel:
        return probas_avg.round().astype(int)
    return np.argmax(probas_avg, axis=1)

