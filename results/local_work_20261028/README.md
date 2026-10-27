# Результаты локального исследования — 28.10.2026

[Методика, лучшие параметры и выводы](../../docs/local-work-research-20261028.md) · [Таблица параметров](best_parameters.csv) · [Контроль](holdout_summary.csv) · [Аудит](audit.json).

Все три класса на одном графике. Montserrat; точки — измерения, PCHIP — только плавное соединение.

## Основные результаты

![Независимый контроль](plots/holdout_quality.png)

[SVG](plots/holdout_quality.svg) · [Сравнение качества](plots/paired_survey_error_pct.png) · [Сравнение времени](plots/paired_survey_seconds.png).

## Число поколений

![Число поколений](plots/ofat_generations_error_pct.png)

[PNG](plots/ofat_generations_error_pct.png) · [SVG](plots/ofat_generations_error_pct.svg) · [Время PNG](plots/ofat_generations_seconds.png) · [Время SVG](plots/ofat_generations_seconds.svg).

## Общая популяция

![Общая популяция](plots/ofat_population_error_pct.png)

[PNG](plots/ofat_population_error_pct.png) · [SVG](plots/ofat_population_error_pct.svg) · [Время PNG](plots/ofat_population_seconds.png) · [Время SVG](plots/ofat_population_seconds.svg).

## RSS: исходный / ранговый

![RSS: исходный / ранговый](plots/ofat_selection_error_pct.png)

[PNG](plots/ofat_selection_error_pct.png) · [SVG](plots/ofat_selection_error_pct.svg) · [Время PNG](plots/ofat_selection_seconds.png) · [Время SVG](plots/ofat_selection_seconds.svg).

## Степень рангового RSS

![Степень рангового RSS](plots/ofat_rank_power_error_pct.png)

[PNG](plots/ofat_rank_power_error_pct.png) · [SVG](plots/ofat_rank_power_error_pct.svg) · [Время PNG](plots/ofat_rank_power_seconds.png) · [Время SVG](plots/ofat_rank_power_seconds.svg).

## Вероятность мутации

![Вероятность мутации](plots/ofat_mutation_prob_error_pct.png)

[PNG](plots/ofat_mutation_prob_error_pct.png) · [SVG](plots/ofat_mutation_prob_error_pct.svg) · [Время PNG](plots/ofat_mutation_prob_seconds.png) · [Время SVG](plots/ofat_mutation_prob_seconds.svg).

## Максимальная длина инверсии, %

![Максимальная длина инверсии, %](plots/ofat_inversion_pct_error_pct.png)

[PNG](plots/ofat_inversion_pct_error_pct.png) · [SVG](plots/ofat_inversion_pct_error_pct.svg) · [Время PNG](plots/ofat_inversion_pct_seconds.png) · [Время SVG](plots/ofat_inversion_pct_seconds.svg).

## Доля элиты

![Доля элиты](plots/ofat_elitism_error_pct.png)

[PNG](plots/ofat_elitism_error_pct.png) · [SVG](plots/ofat_elitism_error_pct.svg) · [Время PNG](plots/ofat_elitism_seconds.png) · [Время SVG](plots/ofat_elitism_seconds.svg).

## Период локального поиска (0 = выключен)

![Период локального поиска (0 = выключен)](plots/ofat_local_every_error_pct.png)

[PNG](plots/ofat_local_every_error_pct.png) · [SVG](plots/ofat_local_every_error_pct.svg) · [Время PNG](plots/ofat_local_every_seconds.png) · [Время SVG](plots/ofat_local_every_seconds.svg).

## Число кандидатов K

![Число кандидатов K](plots/ofat_local_candidates_error_pct.png)

[PNG](plots/ofat_local_candidates_error_pct.png) · [SVG](plots/ofat_local_candidates_error_pct.svg) · [Время PNG](plots/ofat_local_candidates_seconds.png) · [Время SVG](plots/ofat_local_candidates_seconds.svg).

## Проходы 2-swap

![Проходы 2-swap](plots/ofat_local_passes_error_pct.png)

[PNG](plots/ofat_local_passes_error_pct.png) · [SVG](plots/ofat_local_passes_error_pct.svg) · [Время PNG](plots/ofat_local_passes_seconds.png) · [Время SVG](plots/ofat_local_passes_seconds.svg).

## Общее число локальных улучшений за событие

![Общее число локальных улучшений за событие](plots/ofat_local_top_error_pct.png)

[PNG](plots/ofat_local_top_error_pct.png) · [SVG](plots/ofat_local_top_error_pct.svg) · [Время PNG](plots/ofat_local_top_seconds.png) · [Время SVG](plots/ofat_local_top_seconds.svg).

## Тройные циклы 3-cycle

![Тройные циклы 3-cycle](plots/ofat_cycle3_error_pct.png)

[PNG](plots/ofat_cycle3_error_pct.png) · [SVG](plots/ofat_cycle3_error_pct.svg) · [Время PNG](plots/ofat_cycle3_seconds.png) · [Время SVG](plots/ofat_cycle3_seconds.svg).

## Проходы 3-cycle

![Проходы 3-cycle](plots/ofat_cycle3_passes_error_pct.png)

[PNG](plots/ofat_cycle3_passes_error_pct.png) · [SVG](plots/ofat_cycle3_passes_error_pct.svg) · [Время PNG](plots/ofat_cycle3_passes_seconds.png) · [Время SVG](plots/ofat_cycle3_passes_seconds.svg).

## Адаптивная инверсия

![Адаптивная инверсия](plots/ofat_adaptive_error_pct.png)

[PNG](plots/ofat_adaptive_error_pct.png) · [SVG](plots/ofat_adaptive_error_pct.svg) · [Время PNG](plots/ofat_adaptive_seconds.png) · [Время SVG](plots/ofat_adaptive_seconds.svg).

## Горизонт адаптации

![Горизонт адаптации](plots/ofat_adaptive_horizon_error_pct.png)

[PNG](plots/ofat_adaptive_horizon_error_pct.png) · [SVG](plots/ofat_adaptive_horizon_error_pct.svg) · [Время PNG](plots/ofat_adaptive_horizon_seconds.png) · [Время SVG](plots/ofat_adaptive_horizon_seconds.svg).

## Вероятность длинной инверсии

![Вероятность длинной инверсии](plots/ofat_adaptive_long_prob_error_pct.png)

[PNG](plots/ofat_adaptive_long_prob_error_pct.png) · [SVG](plots/ofat_adaptive_long_prob_error_pct.svg) · [Время PNG](plots/ofat_adaptive_long_prob_seconds.png) · [Время SVG](plots/ofat_adaptive_long_prob_seconds.svg).

## Длина дополнительной инверсии, %

![Длина дополнительной инверсии, %](plots/ofat_adaptive_long_pct_error_pct.png)

[PNG](plots/ofat_adaptive_long_pct_error_pct.png) · [SVG](plots/ofat_adaptive_long_pct_error_pct.svg) · [Время PNG](plots/ofat_adaptive_long_pct_seconds.png) · [Время SVG](plots/ofat_adaptive_long_pct_seconds.svg).

## Число островов при равной общей популяции

![Число островов при равной общей популяции](plots/ofat_islands_error_pct.png)

[PNG](plots/ofat_islands_error_pct.png) · [SVG](plots/ofat_islands_error_pct.svg) · [Время PNG](plots/ofat_islands_seconds.png) · [Время SVG](plots/ofat_islands_seconds.svg).

## Интервал кольцевой миграции

![Интервал кольцевой миграции](plots/ofat_migration_interval_error_pct.png)

[PNG](plots/ofat_migration_interval_error_pct.png) · [SVG](plots/ofat_migration_interval_error_pct.svg) · [Время PNG](plots/ofat_migration_interval_seconds.png) · [Время SVG](plots/ofat_migration_interval_seconds.svg).

## Доля мигрантов

![Доля мигрантов](plots/ofat_migration_pct_error_pct.png)

[PNG](plots/ofat_migration_pct_error_pct.png) · [SVG](plots/ofat_migration_pct_error_pct.svg) · [Время PNG](plots/ofat_migration_pct_seconds.png) · [Время SVG](plots/ofat_migration_pct_seconds.svg).

