"""Legacy island demonstration must not contradict equal-total-population task."""
import os
import random
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,os.path.join(os.path.dirname(__file__),"..","lab1"))
import lab1_test_migrations as islands
from lab1_test import run_ga

from lab1 import Employees


class LegacyIslandsTests(unittest.TestCase):
    def test_num_islands_partitions_total_population_instead_of_multiplying(self):
        matrix=[[1,4,2],[3,1,4],[4,2,1]]
        with patch.object(islands,"Employees",wraps=Employees) as factory:
            random.seed(31)
            islands.run_ga_with_migrations(
                population_size=11, mutation_probability=0.7,
                mutation_percent=30, tournament_size=2,
                max_iterations=2, plateau_limit=2,plateau_diff=0,
                migration_interval=3,migration_percent=10,
                num_islands=3,elitist_percent=.2,matrix=matrix)
            populations=[call.kwargs["n"] for call in factory.call_args_list]
        self.assertEqual(populations,[4,4,3])
        self.assertEqual(sum(populations),11)

    def test_one_island_same_result_as_simple_ga_for_same_seed(self):
        matrix=[[1,4,6],[3,1,2],[8,5,1]]
        params={"population_size": 12, "mutation_probability": 0.8,
                    "mutation_percent": 45,"tournament_size": 2,"max_iterations": 8,
                    "plateau_limit": 3,"plateau_diff": 0,"elitist_percent": .2,"matrix": matrix}
        random.seed(777)
        expected=run_ga(**params)
        random.seed(777)
        actual=islands.run_ga_with_migrations(**dict(params,
                        migration_interval=10,migration_percent=10,num_islands=1))
        self.assertEqual(expected,actual)

    def test_plateau_population_mean_is_weighted_by_number_of_individuals(self):
        class Fake:
            def __init__(self,n,fit):
                self.population=[object() for _ in range(n)]
                self.value=fit
            def fitness(self):
                return self.value
        self.assertAlmostEqual(
            islands.weighted_population_fitness([Fake(2,.1),Fake(8,.5)]),
            .42
        )

    def test_invalid_migration_configuration_rejected(self):
        matrix=[[1,10],[10,1]]
        params={"population_size": 5,"mutation_probability": .7,
            "mutation_percent": 40,"tournament_size": 2,"max_iterations": 4,
            "plateau_limit": 2,"plateau_diff": 1e-6,
            "migration_interval": 0,"migration_percent": 20,
            "num_islands": 2,"elitist_percent": .2,"matrix": matrix}
        with self.assertRaises(ValueError):
            islands.run_ga_with_migrations(**params)


if __name__=="__main__":
    unittest.main()
