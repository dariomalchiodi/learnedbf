import numpy as np
import unittest

from learnedbf import *
from learnedbf.BF import BloomFilter, ClassicalBloomFilter


class TestPybloomliveClassicalBloomFilter(unittest.TestCase):

    def _check_train(self, filter_class):
        for num_keys in np.logspace(1, 5, 5).astype(int):
            for epsilon in (0.01, 0.05, 0.1, 0.2):
                X = np.random.randint(0, 1_000_000, size=(num_keys, 1))
                bf = BloomFilter(n=num_keys,
                                          epsilon=epsilon,
                                          filter_class=filter_class)
                bf.fit(X)
                self.assertTrue(bf.predict(X).all())

    def test_train(self):
        for filter_class in (ClassicalBloomFilter, ):
            self._check_train(filter_class)

    def _check_empty(self, filter_class):
        bf = BloomFilter(n=20,
                                  epsilon=0.01,
                                  filter_class=filter_class)

        with self.assertRaises(ValueError):
            bf.fit([[]])

    def test_empty_dataset(self):
        for filter_class in (ClassicalBloomFilter, ):
            self._check_empty(filter_class)

    def _check_score(self, filter_class):
        for num_keys in np.logspace(1, 5, 5).astype(int):
            for epsilon in (0.01, 0.05, 0.1, 0.2):
                X = np.random.randint(0, 1_000_000, size=(num_keys, 1))
                bf = BloomFilter(n=num_keys,
                                          epsilon=epsilon,
                                          filter_class=filter_class)
                bf.fit(X)

                num_queries = 10_000
                Q = np.random.randint(0, 1_000_000, size=(num_queries, 1))
                Q = np.expand_dims(
                    np.array(list(set(Q.squeeze()) - set(X.squeeze()))),
                    axis=1
                )

                # TODO: this is ugly, check with test set in LBF
                self.assertTrue(bf.estimate_FPR(Q) <= epsilon * 2)

    def test_score(self):
        for filter_class in (ClassicalBloomFilter, ):
            self._check_score(filter_class)
    
    def _check_param_consistency(self, filter_class):
        n_values = (10, 100, 1000, 10000)
        epsilon_values = (0.01, 0.05, 0.1, 0.2)
        m_values = (200, 2000, 20000)
        for n in n_values:
            for epsilon in epsilon_values:
                bf = filter_class(n=n, epsilon=epsilon)
                m = math.ceil(-n * np.log(epsilon) / np.log(2)**2)
                self.assertEqual(bf.n, n)

        for epsilon in epsilon_values:
            for m in m_values:
                bf = ClassicalBloomFilter(epsilon=epsilon, m=m)
                n = math.ceil(-m * np.log(2)**2 / np.log(epsilon))
                self.assertTrue(abs(bf.m - m) <= 1)


        n_m_values = ((10, 100), (10, 1000), (100, 2000), (1000, 10000))
        for n, m in n_m_values:
            bf = ClassicalBloomFilter(n=n, m=m)
            epsilon = np.exp(-np.log(2)**2 * m / n)
            if epsilon < 1e-9:
                epsilon = 1e-9
            self.assertEqual(bf.epsilon, epsilon)
        
        with self.assertRaises(ValueError):
            ClassicalBloomFilter(1, 1, 1)
            ClassicalBloomFilter()

    def test_param_consistency(self):
        for filter_class in (ClassicalBloomFilter, ):
            self._check_param_consistency(filter_class)

if __name__ == '__main__':
    unittest.main()
