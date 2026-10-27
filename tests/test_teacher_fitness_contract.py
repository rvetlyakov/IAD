"""Explicit checks for the teacher's initial population/fitness tasks."""
import random
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lab1'))
from hybrid_ga import costs

from lab1 import Employee, Employees


class TeacherFitnessContractTests(unittest.TestCase):
    def test_individual_fitness_matches_assignment_objective(self):
        matrix=[[9,2,8],[5,7,1],[3,6,4]]
        chromosome=[1,2,0]
        individual=Employee(chromosome,3,matrix,50)
        self.assertAlmostEqual(individual.fit,1/6)
        self.assertEqual(int(costs(np.array(matrix,dtype=np.int32),chromosome)[0]),6)

    def test_population_fitness_is_the_arithmetic_mean(self):
        matrix=[[1,9],[8,2]]
        pop=Employees(2,.7,matrix,50,.2)
        pop.population=[Employee([0,1],2,matrix,50),Employee([1,0],2,matrix,50)]
        self.assertAlmostEqual(pop.fitness(),(1/3+1/17)/2)

    def test_generated_population_is_valid_and_evaluated(self):
        random.seed(20261028)
        matrix=[[1,9,9],[9,1,9],[9,9,1]]
        pop=Employees(7,.7,matrix,50,.2)
        self.assertEqual(len(pop.population),7)
        for individual in pop.population:
            self.assertEqual(sorted(individual.vector),[0,1,2])
            cost=sum(matrix[i][individual.vector[i]] for i in range(3))
            self.assertAlmostEqual(individual.fit,1/cost)
