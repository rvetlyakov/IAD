"""Historical name kept for compatibility with early lab notes.

The original copy was NOT executable (undefined Employees/num_islands and
factorial exhaustive search). The maintained implementation is in
lab1_test_migrations.py and uses SciPy's exact assignment solver.
"""
from lab1_test_migrations import run_ga_with_migrations


def run_ga_with_migraions(
    population_size,
    mutation_probability,
    tournament_size,
    max_iterations,
    plateau_limit,
    migration_interval,
    migration_percent,
    matrix,
    *,
    num_islands=3,
    mutation_percent=30.0,
    plateau_diff=1e-6,
    elitist_percent=0.2,
):
    """Legacy misspelled entry point, with a FIXED total population budget.

    New code should use run_ga_with_migrations directly. This demonstration
    is sequential, not multiprocessing; see hybrid_ga.run for real processes.
    """
    return run_ga_with_migrations(
        population_size=population_size,
        mutation_probability=mutation_probability,
        mutation_percent=mutation_percent,
        tournament_size=tournament_size,
        max_iterations=max_iterations,
        plateau_limit=plateau_limit,
        plateau_diff=plateau_diff,
        migration_interval=migration_interval,
        migration_percent=migration_percent,
        num_islands=num_islands,
        elitist_percent=elitist_percent,
        matrix=matrix,
    )
