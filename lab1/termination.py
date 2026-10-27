"""Scale-independent convergence detection for assignment GAs."""
import itertools


def plateau_reached(fitness_history, plateau_limit, plateau_diff):
    """True when the last `plateau_limit` relative fitness changes are small.

    Relative changes are based on the mean fitness from consecutive generations.
    An exact zero threshold only terminates a sequence of unchanged means.
    """
    if plateau_limit < 1 or plateau_diff < 0:
        raise ValueError("plateau_limit must be >= 1 and plateau_diff >= 0")
    if len(fitness_history) < plateau_limit + 1:
        return False
    window = fitness_history[-(plateau_limit + 1):]
    differences = [
        abs(after - before) / max(abs(before), 1e-12)
        for before, after in itertools.pairwise(window)
    ]
    return max(differences) <= plateau_diff
