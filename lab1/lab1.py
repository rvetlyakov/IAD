# Формирование популяции, функцию вычисления приспособленности популяции и алгоритм мутации
# В презентации код и его тестирование
# прийти в 16.10

import math
import copy
from GA_abstract import Vector, Population
import random
import math
from itertools import combinations

class Employee( Vector ):

    def __init__(self, vector, length, matrix, mutation_percent):
        self.vector = copy.deepcopy(vector)
        self.length = length
        self.matrix = matrix
        self.fit = self.fitness(vector) # Поменял местами последние 2 строчки, так как фитнесс использует матрицу
        self.mutation_percent = mutation_percent

    def mutate(self):
        """Return a new chromosome with a randomly reversed contiguous segment.

        mutation_percent is the maximum segment length as a percentage of the
        chromosome length. A useful mutation needs at least two positions.
        """
        if not 0 < self.mutation_percent <= 100:
            raise ValueError("mutation_percent must be in (0, 100]")
        if self.length < 2:
            return Employee(self.vector, self.length, self.matrix, self.mutation_percent)

        max_segment = min(self.length, max(2, math.ceil(
            self.length * self.mutation_percent / 100
        )))
        segment_length = random.randint(2, max_segment)
        start = random.randint(0, self.length - segment_length)
        result = self.vector.copy()
        result[start:start + segment_length] = reversed(
            result[start:start + segment_length]
        )
        return Employee(result, self.length, self.matrix, self.mutation_percent)

    def fitness(self, vector): # pyright: ignore[reportIncompatibleMethodOverride]
        """Evaluate the chromosome passed in, not a stale instance vector.

        Zero-cost assignments are valid; map them to large finite fitness
        so RSS remains numerically defined. Costs below zero are unsupported
        by this non-negative assignment-cost formulation.
        """
        cost = sum(self.matrix[i][vector[i]] for i in range(self.length))
        if cost < 0:
            raise ValueError("Assignment costs must sum to a non-negative value")
        return 1.0 / max(float(cost), 1e-12)


    

class Employees( Population ):

    """Класс для решения задачи о расстановке сотрудников с помощью ГА
    @params: population_size \n
    n - размер популяции \n
    mutation_probability - вероятность мутации для каждого вектора. Должна быть высокой. \n
    matrix - матрица стоимостей наёма i-го сотрудника на j-е место \n
    t - количество участников одного раунда турнирного отбора\n
    mutation_percent - процент от длины вектора, который переворачивается\n
    elitist_percent - процент особей, который сохраняется"""

    def __init__(self, n, mutation_probability, matrix, mutation_percent, elitist_percent, t=2):
        self.matrix = matrix
        self.population = []
        self.n = n
        self.mutation_probability = mutation_probability
        self.form_first_population(mutation_percent)
        self.elitist_percent = elitist_percent
        # self.matrix = matrix Поменял местами, так как суперкласс вызывает form_first_population, который вызывает self.matrix
        self.t = t
        self.mutation_percent = mutation_percent
        

    def form_first_population(self, mutation_percent):
        self.population = []
        chromosome_len = len(self.matrix)
        numbers = range(chromosome_len)
        while len(self.population) < self.n:
            #self.population.append(random.sample(numbers, k=self.n))
            # Список чисел нужно преобразовать к классу employee

            random_vector = random.sample(numbers, k=chromosome_len)
            new_employee = Employee(random_vector, chromosome_len, self.matrix, mutation_percent)
            self.population.append(new_employee)

    def select(self):
        """Elitism followed by remainder stochastic sampling (RSS).

        Every selection produces exactly self.n chromosomes. Guaranteed
        copies use the integer parts of expected counts; the remaining
        positions are sampled from normalized fractional remainders.
        """
        if self.n < 1 or not self.population:
            raise ValueError("Population must be non-empty and n >= 1")
        if not 0 <= self.elitist_percent <= 1:
            raise ValueError("elitist_percent must be in [0, 1]")

        elite_count = int(self.n * self.elitist_percent)
        ranked = sorted(self.population, key=lambda v: v.fit, reverse=True)
        new_population = ranked[:elite_count]
        remaining = self.n - elite_count

        if remaining:
            total_fitness = sum(v.fit for v in self.population)
            if total_fitness <= 0 or any(v.fit < 0 for v in self.population):
                raise ValueError("RSS requires positive total, non-negative fitness")

            expected = [remaining * v.fit / total_fitness for v in self.population]
            for individual, copies in zip(self.population, expected):
                for _ in range(math.floor(copies)):
                    new_population.append(
                        Employee(individual.vector, len(self.matrix),
                                 self.matrix, self.mutation_percent)
                    )

            missing = self.n - len(new_population)
            if missing:
                fractional = [x - math.floor(x) for x in expected]
                # Rounding can theoretically exhaust fractions; use an
                # unbiased uniform fallback for that degenerate case.
                weights = fractional if sum(fractional) > 0 else None
                for individual in random.choices(self.population, weights=weights, k=missing):
                    new_population.append(
                        Employee(individual.vector, len(self.matrix),
                                 self.matrix, self.mutation_percent)
                    )

        self.population = new_population

    def mutate(self):
        """Функция вместо кроссовера"""

        # mutants = 
        mutants = []
        for vector in self.population:
            num = random.random()
            if num < self.mutation_probability:
                new = vector.mutate()
                mutants.append(new)  # Добавляем во временный список

        self.population.extend(mutants)

    def crossover(self):
        pass

    def fitness(self):
        """Функция приспособленности популяции. Записывает приспособленность в переменную класса self.fit и возвращает её"""

        self.fit = sum(v.fit for v in self.population) / len(self.population)
        return self.fit

    def get_best(self):
        """Вспомогательный метод для поиска лучшей особи в популяции"""
        best = max(self.population, key=lambda v: v.fit)

        cost = sum(self.matrix[i][best.vector[i]] for i in range(len(self.matrix)))
        return best.vector, cost

    def get_top(self, k) -> list[Employee]:
        """
        BRIEF: функция, отбирающая k% лучших особей
        PARAMS:\n
        k - процент лучших особей от популяции, число от 1 до 100\n
        RETURNS: list[Employee] - список лучших объектов
        """

        top_number = math.ceil(self.n * k / 100)
        return sorted(self.population, key=lambda v: v.fit, reverse=True)[:top_number]
        #добавил реверс и ограничение до топ намбер

    def extend_population(self, addition: list[Employee]) -> None:
        """BRIEF: функция, расширяющая популяцию выбранными векторами
        \nPARAMS: addition: list[Employee] - список с добавляемыми векторами"""
        self.population.extend(addition)
