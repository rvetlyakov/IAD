import math
from lab1 import Employees
from itertools import permutations
from scipy.optimize import linear_sum_assignment
from termination import plateau_reached


matrices = {
    "ideal_matrice": [
        [1, 100, 100, 100, 100],
        [100, 1, 100, 100, 100],
        [100, 100, 1, 100, 100],
        [100, 100, 100, 1, 100],
        [100, 100, 100, 100, 1]
    ],
    "mat1": [[10, 19, 8, 15],
               [13, 11, 17, 10],
               [12, 14, 16, 18],
               [19, 8, 11, 14]]
    ,
    "mat2": [[5, 12, 18, 20, 23, 14],
               [10, 9, 6, 12, 15, 11],
               [14, 17, 12, 8, 11, 16],
               [21, 15, 13, 16, 9, 12],
               [12, 10, 14, 18, 22, 7],
               [8, 13, 16, 11, 15, 10]]
    ,
    "mat3": [[17, 3, 14, 8, 1, 19, 5, 12],
             [11, 20, 2, 15, 7, 18, 13, 5],
             [8, 16, 4, 11, 1, 14, 19, 2],
             [7, 13, 6, 17, 38, 12, 15, 9],
             [20, 4, 10, 51, 14, 6, 18, 3],
             [11, 16, 71, 2, 13, 5, 8, 12],
             [4, 20, 17, 7, 10, 15, 1, 14],
             [6, 12, 4, 64, 53, 3, 34, 43]]
}

NUM_ISLANDS = 3  # Количество независимых популяций (островов)
POPULATION_SIZE = 150  # Общее число особей ВСЕХ островов (50 × 3)
MUTATION_PROBABILITY = 0.7
MUTATION_PERCENT = 30 # процент от длины элемента вектора, который переворачивается в ходе мутации
TOURNAMENT_SIZE = 2
MAX_ITERATIONS = 1000
PLATEAU_LIMIT = 3
PLATEAU_DIFF = 2e-4
ELITIST_PERCENT = 0.2

MIGRATION_INTERVAL = 10  # Каждые 10 поколений происходит обмен
MIGRATION_PERCENT = 10  # Передаем 10% лучших особей (5 штук при размере 50)

def weighted_population_fitness(islands):
    """Mean fitness of *all individuals*, not mean of island means.

    When total population does not divide evenly by island count, islands have
    different sizes. An unweighted mean of means is then mathematically wrong.
    """
    total = sum(len(ga.population) for ga in islands)
    if total <= 0:
        raise ValueError("No individuals across island populations")
    return sum(ga.fitness() * len(ga.population) for ga in islands) / total


def run_ga_with_migrations(
        population_size,
        mutation_probability,
        mutation_percent,
        tournament_size,
        max_iterations,
        plateau_limit,
        plateau_diff,
        migration_interval,
        migration_percent,
        num_islands,
        elitist_percent,
        matrix
):
    """Sequential island demonstrator using a **total** population budget.

    total population_size is split as evenly as possible among num_islands.
    This module is a SEQUENTIAL teaching example; for ACTUAL multiprocessing
    and equal-generation comparison use scale_benchmark.run_variant().
    Both preserve best visited chromosome cost, including lost mutants.
    """
    if population_size < 1 or num_islands < 1 or population_size < num_islands:
        raise ValueError("Need at least one individual per island")
    if migration_interval < 1:
        raise ValueError("migration_interval must be positive")
    if not 0 <= migration_percent <= 100:
        raise ValueError("migration_percent must be in [0, 100]")
    per_island, extra = divmod(population_size, num_islands)
    sizes = [per_island + (i < extra) for i in range(num_islands)]

    islands = [
        Employees(n=size,
                  mutation_probability=mutation_probability,
                  matrix=matrix,
                  mutation_percent=mutation_percent,
                  elitist_percent=elitist_percent,
                  t=tournament_size)
        for size in sizes
    ]

    best_so_far = min(ga.get_best()[1] for ga in islands)
    iteration = 0
    fitness_history = [weighted_population_fitness(islands)]

    while iteration < max_iterations:
        for ga in islands:
            ga.mutate()
            best_so_far = min(best_so_far, ga.get_best()[1])
            ga.select()
            best_so_far = min(best_so_far, ga.get_best()[1])
        iteration += 1

        if num_islands > 1 and iteration % migration_interval == 0:
            migrants_from = [ga.get_top(migration_percent) for ga in islands]
            for i in range(num_islands):
                islands[(i + 1) % num_islands].extend_population(migrants_from[i])
            best_so_far = min(best_so_far, *(ga.get_best()[1] for ga in islands))
            for ga in islands:
                ga.select()
                best_so_far = min(best_so_far, ga.get_best()[1])

        fitness_history.append(weighted_population_fitness(islands))
        if plateau_reached(fitness_history, plateau_limit, plateau_diff):
            break

    row_ind, col_ind = linear_sum_assignment(matrix)
    exact = sum(matrix[int(i)][int(j)] for i, j in zip(row_ind, col_ind))
    if exact <= 0:
        # A zero-cost optimum is valid. The absolute gap is meaningful here;
        # a relative gap with zero denominator is mathematically undefined.
        return (best_so_far == exact, iteration, 0 if best_so_far == 0 else float("inf"))
    return (best_so_far == exact, iteration,
            (best_so_far - exact) / exact)


if __name__ == "__main__":
    for test_name, matrix in matrices.items():
        # print(f"\n{'=' * 10} {test_name} (Мультипопуляционный ГА) {'=' * 10}")
        run_ga_with_migrations(POPULATION_SIZE,
                              MUTATION_PROBABILITY,
                              MUTATION_PERCENT,
                              TOURNAMENT_SIZE,
                              MAX_ITERATIONS,
                              PLATEAU_LIMIT,
                              PLATEAU_DIFF,
                              MIGRATION_INTERVAL,
                              MIGRATION_PERCENT,
                              NUM_ISLANDS,
                              ELITIST_PERCENT,
                              matrix)

    # count = 0
    # iterations = 0
    # for i in range(100):
    #     result = run_ga_with_migraions(POPULATION_SIZE,
    #                                 MUTATION_PROBABILITY,
    #                                 TOURNAMENT_SIZE,
    #                                 MAX_ITERATIONS,
    #                                 PLATEAU_LIMIT,
    #                                 PLATEAU_DIFF,
    #                                 MIGRATION_INTERVAL,
    #                                 MIGRATION_PERCENT,
    #                                 NUM_ISLANDS,
    #                                 matrices["ideal_matrice"])
    #     count += int(result[0])
    #     iterations += result[1]

    # print(f'На "идеальных" данных алгоритм отработал правильно в {count}% случаев.\n',
    #       f'В среднем это заняло {iterations // 100} итераций')
        