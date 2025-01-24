import numpy as np
import math
import unittest
from learnedbf.classifiers import ScoredDecisionTreeClassifier, get_tree_size
from learnedbf import LearnedBloomFilter, SandwichedLearnedBloomFilter

learned_filter_classes = [LearnedBloomFilter]

class TestScoredDT(unittest.TestCase):

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

                scl = ScoredDecisionTreeClassifier(
                    random_state=42)
                scl.fit(objects, labels)

                for filter_class in learned_filter_classes:
                    lbf = filter_class(classifier=scl,
                                            epsilon=0.1,
                                            n=len(objects))
                    lbf.fit(objects, labels)

                    self.assertTrue(not lbf.predict(objects[:mid]).all())
                    self.assertTrue(lbf.predict(objects[mid:]).all())

    def test_consistent_initial_filter(self):

        n_samples = [50,100]
        Fp = 0.2

        for n in n_samples:
            objects = np.expand_dims(np.arange(0, n*2), axis=1)
            labels_f = np.array([False] * n)
            labels_t = np.array([True] * n)
            #we force a fraction of FP
            FP_mask = np.random.random(size=labels_t.shape[0]) < Fp
            labels_t[FP_mask] = ~labels_t[FP_mask]
            labels = np.concatenate((labels_f, labels_t))

            scl = ScoredDecisionTreeClassifier(max_depth=2)
            scl.fit(objects, labels)
            slbf = SandwichedLearnedBloomFilter(epsilon=0.1)
            slbf.fit(objects, labels)

            self.assertIsNotNone(slbf.initial_filter_)
            self.assertEqual(sum(labels), slbf.initial_filter_.n)


    def test_get_size(self):
        n = 100

        for dim in [1, 3, 5, 10, 35]:
            objects = np.random.random(size=(n, dim))

            # we force the dataset to be linearly separable in the first dimension
            # objects[:, 0] = np.arange(0, n)
            labels = [False] * int(math.floor(0.4 * n)) +  \
                    [True] * int(math.ceil(0.6 * n))

            dtcl = ScoredDecisionTreeClassifier()
            dtcl.fit(objects, labels)

            for filter_class in learned_filter_classes:
                lbf = filter_class(classifier=dtcl,
                                   epsilon=0.1,
                                   n=len(objects))
                lbf.fit(objects, labels)

                self.assertTrue(dtcl.get_size() == get_tree_size(dtcl))

                lbf_size = dtcl.get_size()
                if lbf.backup_filter_ is not None:
                    lbf_size += lbf.backup_filter_.get_size()

                self.assertTrue(sum(lbf.get_size().values()) == lbf_size)

                self.assertTrue(dtcl.get_size() == lbf.classifier.get_size())

if __name__ == '__main__':
    unittest.main()

        # add tests for get_size in all classifiers, complete tests for DT e RF
