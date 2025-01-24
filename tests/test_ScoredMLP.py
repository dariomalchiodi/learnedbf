
import math

import numpy as np
import unittest
from learnedbf.classifiers import ScoredMLP
from learnedbf.BF import BloomFilter
from learnedbf import LearnedBloomFilter, SandwichedLearnedBloomFilter

learned_filter_classes = [LearnedBloomFilter]


class TestScoredMLP(unittest.TestCase):

    def test_linear_separability(self):
        from_ = 1

        for to in [5, 10, 100, 10000]:
            mid = to *2 // 3 # last negative item
            objects = np.expand_dims(np.arange(from_, to), axis=1)

            labels = [False] * (mid - from_ + 1) +  [True] * (to - mid - 1)

            scl = ScoredMLP(hidden_layer_sizes=(5,), max_iter=5000, random_state=42)
            scl.fit(objects, labels)

            for filter_class in learned_filter_classes:
                lbf = filter_class(classifier=scl,
                                   epsilon=0.1,
                                   n=len(objects))
                lbf.fit(objects, labels)

                self.assertTrue(not lbf.predict(objects[:mid]).all())
                self.assertTrue(lbf.predict(objects[mid:]).all())

    def test_consistent_backup_filter(self):
        from_ = 1

        for to in [5, 10, 100, 10000]:
            mid = to * 2 // 3 # last negative item


            for num_FN in (1, 3, 10, 100, 1000):
                if num_FN >= mid:
                    break

                objects = np.expand_dims(np.arange(from_, to), axis=1)

                labels = [False] * (mid - from_ + 1) +  [True] * (to - mid - 1)

                labels[:num_FN] = [True] * num_FN

                scl = ScoredMLP(hidden_layer_sizes=(5,), max_iter=5000,
                                random_state=42)
                scl.fit(objects, labels)

                for filter_class in learned_filter_classes:
                    lbf = filter_class(classifier=scl,
                                            epsilon=0.1,
                                            n=len(objects))
                    lbf.fit(objects, labels)

                    self.assertIsNotNone(lbf.backup_filter_)

                    self.assertTrue(not lbf.predict(objects[:mid]).all())
                    self.assertTrue(lbf.predict(objects[mid:]).all())

                    self.assertTrue(lbf.backup_filter_.n == num_FN)

    def test_consistent_initial_filter(self):

        n_samples = [50, 500]
        Fp = 0.2

        for n in n_samples:
            objects = np.expand_dims(np.arange(0, n*2), axis=1)
            labels_f = np.array([False] * n)
            labels_t = np.array([True] * n)
            #we force a fraction of FP
            FP_mask = np.random.random(size=labels_t.shape[0]) < Fp
            labels_t[FP_mask] = ~labels_t[FP_mask]
            labels = np.concatenate((labels_f, labels_t))

            scl = ScoredMLP(random_state=42,
                                activation="logistic",
                                hidden_layer_sizes=(5,),
                                max_iter=10000)
            scl.fit(objects, labels)
            slbf = SandwichedLearnedBloomFilter(epsilon=0.1)
            slbf.fit(objects, labels)

            self.assertIsNotNone(slbf.initial_filter_)
            self.assertEqual(sum(labels), slbf.initial_filter_.n)

    def test_get_size(self):
        num_examples = 10

        configurations = [[10], [7, 2], [50, 20, 7, 2],
                        [9, 2, 9], [34, 8]]

        for dim in [1, 3, 5, 10, 35]:
            objects = np.random.random(size=(num_examples, dim))

            # we force the dataset to be linearly separable in the first dimension
            objects[:, 0] = np.arange(0, num_examples)
            labels = [False] * int(math.floor(0.4 * num_examples)) +  \
                    [True] * int(math.ceil(0.6 * num_examples))

            for hidden_layer_sizes in configurations:
                scl = ScoredMLP(random_state=42,
                                hidden_layer_sizes=hidden_layer_sizes,
                                max_iter=10000)
                scl.fit(objects, labels)

                for filter_class in learned_filter_classes:
                    lbf = filter_class(classifier=scl,
                                       epsilon=0.1,
                                       n=len(objects))
                    lbf.fit(objects, labels)

                    start = np.array([dim] + hidden_layer_sizes) + 1
                    end = np.array(hidden_layer_sizes + [1])
                    num_connections = sum([s*e for s, e in zip(start, end)])
                    classifier_size = num_connections * scl.float_size

                    lbf_size = classifier_size
                    if lbf.backup_filter_ is not None:
                        lbf_size += lbf.backup_filter_.get_size()

                    self.assertTrue(scl.get_size() == classifier_size)
                    self.assertTrue(sum(lbf.get_size().values()) == lbf_size)


if __name__ == '__main__':
    unittest.main()
