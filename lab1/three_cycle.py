"""Optional three-worker cyclic improvement for permutation assignments.

Not an exact assignment solver. Examines at most K cheap target columns
for each of two affected workers and only accepts strict cost decreases.
"""
import numpy as np


def improve_three_cycle(matrix, permutation, candidate_columns, passes=2):
    a = np.asarray(matrix)
    v = np.asarray(permutation, dtype=np.int32).copy()
    n = len(v)
    inverse = np.empty(n, dtype=np.int32)
    inverse[v] = np.arange(n)
    changes_total = 0
    for _ in range(passes):
        changes = 0
        for i in range(n):
            col_i = int(v[i])
            best_delta = 0
            best_pair = None
            for target in candidate_columns[i]:
                j = int(inverse[target])
                if i == j:
                    continue
                col_j = int(v[j])
                partial = int(a[i, col_j]) - int(a[i, col_i]) - int(a[j, col_j])
                for second_target in candidate_columns[j]:
                    k = int(inverse[second_target])
                    if i == k or j == k:
                        continue
                    col_k = int(v[k])
                    delta = (partial + int(a[j, col_k])
                             + int(a[k, col_i]) - int(a[k, col_k]))
                    if delta < best_delta:
                        best_delta = delta
                        best_pair = (j, k)
            if best_pair is not None:
                j, k = best_pair
                col_i, col_j, col_k = int(v[i]), int(v[j]), int(v[k])
                v[i], v[j], v[k] = col_j, col_k, col_i
                inverse[col_j], inverse[col_k], inverse[col_i] = i, j, k
                changes += 1
        changes_total += changes
        if changes == 0:
            break
    return v, changes_total
