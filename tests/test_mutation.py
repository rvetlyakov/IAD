import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab1"))
from lab1 import Employee


def individual(n, percent=45):
    return Employee(list(range(n)), n, [[1] * n for _ in range(n)], percent)


class InversionMutationTests(unittest.TestCase):
    def test_one_gene_is_unchanged(self):
        self.assertEqual(individual(1).mutate().vector, [0])

    def test_two_genes_always_swap(self):
        for seed in range(50):
            random.seed(seed)
            self.assertEqual(individual(2).mutate().vector, [1, 0])

    def test_inversion_preserves_valid_permutations(self):
        for n in (2, 3, 5, 10, 15, 25):
            for percent in (15, 30, 45, 60, 100):
                for seed in range(25):
                    with self.subTest(n=n, percent=percent, seed=seed):
                        random.seed(seed)
                        x = individual(n, percent)
                        mutation = x.mutate().vector
                        self.assertEqual(sorted(mutation), x.vector)
                        self.assertNotEqual(mutation, x.vector)
                        self.assertEqual(x.vector, list(range(n)))

    def test_segment_length_does_not_exceed_percentage_maximum(self):
        for seed in range(60):
            random.seed(seed)
            v = individual(20, 15).mutate().vector
            self.assertLessEqual(sum(i != x for i, x in enumerate(v)), 3)

    def test_invalid_percent_is_rejected(self):
        for invalid in (0, -1, 101):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                individual(4, invalid).mutate()


if __name__ == "__main__":
    unittest.main()
