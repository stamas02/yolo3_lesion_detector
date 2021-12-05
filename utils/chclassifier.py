from scipy.stats import multivariate_normal
import numpy as np


class CHClassifier():
    def __init__(self):
        self._iter = 0
        self._std = None
        self._mean = None
        self.mvn = None
        self.classes = {}
        pass

    def _add_class(self, c):
        if c not in self.classes.keys():
            self.classes[c] = {"data": [],
                               "mvn": []}

    def update(self, img, c):
        self._add_class(c)

        self.classes[c]["data"].append(np.mean(img, axis=(0, 1)))
        mean = np.mean(np.array(self.classes[c]["data"]), axis=0)
        var = np.var(np.array(self.classes[c]["data"]), axis=0)
        I = np.identity(3)

        if len(self.classes[c]["data"]) > 1:
            self.classes[c]["mvn"] = multivariate_normal(mean, I*var)

    def get_classes(self):
        return list(self.classes.keys())

    def likelihood(self, img):
        x = np.mean(img, axis=(0, 1))
        l = []
        for c in self.classes.keys():
            l.append(self.classes[c]["mvn"].pdf(x))
        return np.array(l)

    def posterior(self,img):
        l = self.likelihood(self, img)
        return l/np.sum(l)