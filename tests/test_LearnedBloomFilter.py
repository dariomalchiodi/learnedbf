import unittest
import numpy as np
from learnedbf import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


class TestLBF(unittest.TestCase):
    def setUp(self):
        self.lbf = LBF(n=499, epsilon=0.1,
            model_selection_method=StratifiedKFold(n_splits=2))
        self.lbf1 = LBF(n=499, epsilon=0.1,
            model_selection_method=StratifiedKFold(n_splits=2))
        self.lbf2 = LBF(n=499, epsilon=0.1,
            classifier=RandomForestClassifier(),
            model_selection_method=StratifiedKFold(n_splits=2))
        self.lbf3 = LBF(n=499, epsilon=0.1,
            classifier=KNeighborsClassifier(n_neighbors=1),
            model_selection_method=StratifiedKFold(n_splits=2))
        self.lbf4 = LBF(n=499, epsilon=0.1,
            classifier=KNeighborsClassifier(n_neighbors=2),
            model_selection_method=StratifiedKFold(n_splits=2))

        Xx = np.random.randint(low=0, high=10000, size=1000)
        self.X = []
        self.y = []
        for x in Xx:
            self.X.append([x])
            self.y.append(len(self.X) > 500)
        self.X = np.array(self.X)
        self.y = np.array(self.y)

        self.lbf1.fit(self.X, self.y)
        self.lbf2.fit(self.X, self.y)
        self.lbf3.fit(self.X, self.y)
        self.lbf4.fit(self.X, self.y)

    def test_fit(self):
        assert self.lbf1.is_fitted_
        assert self.lbf2.is_fitted_
        assert self.lbf3.is_fitted_
        assert self.lbf4.is_fitted_

    def test_score(self):

        Xscore = np.random.randint(low=0, high=10000, size=1000)
        X = []
        for x in Xscore:
            X.append([x])
        X = np.array(X)

        self.assertRaises(ValueError, self.lbf.score, X)
        self.assertTrue(abs(self.lbf1.score(X) - self.lbf1.epsilon) < 1E-1)
        self.assertTrue(abs(self.lbf2.score(X) - self.lbf2.epsilon) < 1E-1)
        self.assertTrue(abs(self.lbf3.score(X) - self.lbf3.epsilon) < 1E-1)
        self.assertTrue(abs(self.lbf4.score(X) - self.lbf4.epsilon) < 1E-1)


if __name__ == '__main__':
    unittest.main()
