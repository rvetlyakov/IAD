"""Input contracts for assignment GA: reject invalid hyperparameters early."""
import os, sys, unittest
sys.path.insert(0,os.path.join(os.path.dirname(__file__),"..","lab1"))
from lab1 import Employees


class ConfigurationValidationTests(unittest.TestCase):
    def test_invalid_mutation_probabilities_are_rejected(self):
        matrix=[[1,2],[2,1]]
        for prob in (-0.1,1.1,float("nan")):
            with self.subTest(prob=prob):
                with self.assertRaises(ValueError):
                    Employees(n=5,mutation_probability=prob,matrix=matrix,
                              mutation_percent=30,elitist_percent=.2)

    def test_invalid_elitism_and_segment_percent_rejected_before_run(self):
        matrix=[[1,2],[2,1]]
        for elitism in (-.1,1.2,float("nan")):
            with self.subTest(elitism=elitism):
                with self.assertRaises(ValueError):
                    Employees(5,.7,matrix,30,elitism)
        for percentage in (0,-5,101,float("nan")):
            with self.subTest(percentage=percentage):
                with self.assertRaises(ValueError):
                    Employees(5,.7,matrix,percentage,.2)

    def test_bad_population_and_matrix_dimensions_are_rejected(self):
        for n,matrix in [(0,[[1,2],[2,1]]),
                         (3,[]),
                         (3,[[1,2],[3,4],[5,6]]),
                         (3,[[1,2,3],[2],[3,4,5]])]:
            with self.subTest(n=n,shape=str(matrix)[:40]):
                with self.assertRaises(ValueError):
                    Employees(n,.7,matrix,45,.2)

    def test_small_valid_scenarios_remain_supported(self):
        for length in (1,2,3):
            matrix=[[1 if i==j else 30 for j in range(length)]
                    for i in range(length)]
            instance=Employees(5,0,matrix,20,0)
            self.assertEqual(len(instance.population),5)

if __name__=="__main__":
    unittest.main()
