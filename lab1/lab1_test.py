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

NUM_evolution = 3  # Количество независимых популяций (островов)
POPULATION_SIZE = 250  # Размер каждой популяции
MUTATION_PROBABILITY = 0.7
MUTATION_PERCENT = 40 # процент от длины элемента вектора, который переворачивается в ходе мутации
TOURNAMENT_SIZE = 2
MAX_ITERATIONS = 100
PLATEAU_LIMIT = 5
PLATEAU_DIFF = 1e-16
ELITIST_PERCENT = 0.2

MIGRATION_INTERVAL = 10  # Каждые 10 поколений происходит обмен
MIGRATION_PERCENT = 10  # Передаем 10% лучших особей (5 штук при размере 50)

def run_ga(
        population_size,
        mutation_probability,
        mutation_percent,
        tournament_size,
        max_iterations,
        plateau_limit,
        plateau_diff,
        elitist_percent,
        matrix,
        reference_cost=None
):
    """Функция ГА.\n
**Параметры**:
population_size - размер популяции \n
mutation_probability - вероятность мутации \n
tournament_size - количество участников турнира\n
max_iterations - максимальное количество итераций\n
plateau_limit - длина участка, при достижении постоянства на котором, мы останавливаем алгоритм\n
plateau_diff - максимальное колебание целевой функции, при которой участок считается постоянным\n
matrix - матрица, для которой требуется найти решение\n\n

**Возвращает**: кортеж (found_best, iteration, distance)\n
found_best - boolean - нашел ли алгоритм оптимальное решение\n
iteration - integer - сколько итераций потребовалось для нахождения решения \n
distance - integer - разность между нашим и оптимальным значением."""
    
    evolution = Employees(n=population_size, 
                          mutation_probability=mutation_probability, 
                          matrix=matrix, 
                          mutation_percent=mutation_percent, 
                          elitist_percent = elitist_percent,
                          t=tournament_size)

    iteration = 0
    # One history entry before the first generation: plateau_limit counts
    # actual transitions rather than requiring an extra generation.
    fitness_history = [evolution.fitness()]
    _, best_cost = evolution.get_best()

    # === ГЛАВНЫЙ ЦИКЛ ЭВОЛЮЦИИ ===
    while iteration < max_iterations:

        # Независимая эволюция на каждом острове

        evolution.mutate()  # Мутация
        # Preserve the best *visited* solution, including a mutant that RSS
        # later drops. This matches the scale benchmark's score definition.
        _, trial_cost = evolution.get_best()
        best_cost = min(best_cost, trial_cost)
        evolution.select()  # Отбор лучших
        _, trial_cost = evolution.get_best()
        best_cost = min(best_cost, trial_cost)
        iteration += 1

        # Считаем среднюю приспособленность
        current_pop_fitness = evolution.fitness()  # Считаем среднюю приспособленность популяции
        fitness_history.append(current_pop_fitness)
        if plateau_reached(fitness_history, plateau_limit, plateau_diff):
            break

    # else:
        # print(f"-> Достигнут лимит в {max_iterations} итераций.")

    # 2. Ищем наилучшее решение
    best_vector, _ = evolution.get_best()

    # === НАХОЖДЕНИЕ ЭТАЛОНА ===
    # reference_cost may be calculated once per matrix, outside a GA timer.
    if reference_cost is None:
        rows, cols = linear_sum_assignment(matrix)
        best_exact_cost = sum(matrix[int(i)][int(j)]
                              for i, j in zip(rows, cols))
    else:
        best_exact_cost = reference_cost

    # Печатаем результаты сравнения
    # print(f"Результат Генетического Алгоритма (Лучший остров):")
    # print(f"  - Распределение задач (индексы): {best_vector}")
    # print(f"  - Итоговая стоимость: {best_cost}")

    # print(f"Результат точного метода:")
    # print(f"  - Распределение задач (индексы): {list(best_exact_vector)}")
    # print(f"  - Итоговая стоимость: {best_exact_cost}")

    # print(len(evolution.population), sep="\n")

    if best_cost == best_exact_cost:
        # print("Мультипопуляционный алгоритм нашел точное решение.")
        return (True, iteration, 0)
    else:
        # print(f"Найдено субоптимальное решение. Разница с идеалом: {best_cost - best_exact_cost}")
        return (False, iteration,
                float("inf") if best_exact_cost == 0
                else (best_cost - best_exact_cost) / best_exact_cost)


if __name__ == "__main__":
    for test_name, matrix in matrices.items():
        print(f"\n{'=' * 10} {test_name} (Генетический алгоритм) {'=' * 10}")
        run_ga(POPULATION_SIZE,
                              MUTATION_PROBABILITY,
                              MUTATION_PERCENT,
                              TOURNAMENT_SIZE,
                              MAX_ITERATIONS,
                              PLATEAU_LIMIT,
                              PLATEAU_DIFF,
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
    #                                 NUM_evolution,
    #                                 matrices["ideal_matrice"])
    #     count += int(result[0])
    #     iterations += result[1]

    # print(f'На "идеальных" данных алгоритм отработал правильно в {count}% случаев.\n',
    #       f'В среднем это заняло {iterations // 100} итераций')
        