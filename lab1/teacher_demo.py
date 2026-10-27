"""Concrete source matrices, legacy fitness functions, and current GA demo."""
import csv
import json
import random
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from final_research import configure
from hybrid_ga import Config, run
from local_research import problem, seed_for

from lab1 import Employees


def main():
    root=Path('results/teacher_requirements_audit_20261028');root.mkdir(exist_ok=True)
    selected=json.loads(Path('results/final_offspring_20261028/selected.json').read_text())
    configure()
    n=25;seed=seed_for('holdout',n,0)
    a,exact=problem('holdout',n,0)
    cfg=Config(**selected['small']['single'])
    # Parameters and study constants are output BEFORE invoking either GA.
    constants={'matrix_size':n,'matrix_seed':seed,'ga_seed':seed+17,
               'cost_min':1,'cost_max':100,'objective':'minimize sum of assigned costs',
               'fitness':'individual=1/cost; population=mean(individual fitness)',
               'stopping':'fixed generation budget; plateau is legacy only',
               'crossover':'none, as in the original assignment'}
    parameters={'algorithm_parameters':asdict(cfg),'input_and_study_constants':constants,
                'legacy_fitness_demo':{'population':6,'mutation_probability':.7,
                                       'mutation_percent':50,'elitism':.2,'seed':20261028},
                'diagonal_demo':{'size':5,'diagonal_value':1,'other_values':100,
                                 'ga_seed':20261028,'islands':[1,2]}}
    (root/'DEMO_PARAMETERS.json').write_text(json.dumps(parameters,indent=2))
    print(json.dumps(parameters,indent=2),flush=True)
    np.savetxt(root/'actual_holdout_matrix_25.csv',a,fmt='%d',delimiter=',')
    random.seed(20261028)
    pop=Employees(6,.7,a.tolist(),50,.2)
    records=[]
    for i,individual in enumerate(pop.population):
        cost=int(a[np.arange(n),individual.vector].sum())
        assert abs(individual.fit-1/cost)<1e-12
        records.append({'individual':i,'permutation':json.dumps(individual.vector),
                        'cost':cost,'fitness':individual.fit})
    expected=sum(row['fitness'] for row in records)/len(records)
    assert abs(pop.fitness()-expected)<1e-12
    original=pop.population[0]
    mutated=original.mutate()
    assert sorted(mutated.vector)==list(range(n))
    with (root/'initial_population_fitness.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    result=run(a,cfg,seed+17)
    with Path('results/final_offspring_20261028/holdout.csv').open() as f:
        row=next(r for r in csv.DictReader(f) if r['class_name']=='small' and r['n']=='25' and r['repeat']=='0' and r['variant']=='single')
    assert result['cost']==int(row['cost'])
    diagonal=np.full((5,5),100,dtype=np.int32);np.fill_diagonal(diagonal,1)
    np.savetxt(root/'diagonal_ones_5.csv',diagonal,fmt='%d',delimiter=',')
    checks=[]
    for islands in (1,2):
        ans=run(diagonal,replace(cfg,islands=islands),seed=20261028,workers=2)
        assert ans['cost']==5 and sorted(ans['best'])==list(range(5))
        checks.append({'islands':islands,'cost':ans['cost'],'exact':5,'best':ans['best'].tolist()})
    report={'source':'actual final HOLDOUT, n=25, repeat=0; no new tuning',
            'exact':exact,'ga_cost':result['cost'],'permutation':result['best'].tolist(),
            'error_pct':100*(result['cost']-exact)/exact,
            'population_fitness':pop.fitness(),'manual_population_fitness':expected,
            'mutation_before':original.vector,'mutation_after':mutated.vector,
            'diagonal_checks':checks}
    (root/'demo_results.json').write_text(json.dumps(report,indent=2))
    (root/'DEMO.md').write_text('# Исходные данные и тестирование для презентации\n\n'
        '1. До запуска перечислить параметры из DEMO_PARAMETERS.json.\n'
        '2. Показать actual_holdout_matrix_25.csv: это настоящая матрица финального контроля, а не новый обучающий пример.\n'
        '3. Показать initial_population_fitness.csv: допустимые перестановки, стоимость, приспособленность каждой особи; среднее совпадает с функцией приспособленности популяции.\n'
        '4. Показать инверсию до/после из demo_results.json и сохранение допустимости.\n'
        '5. Показать diagonal_ones_5.csv: на диагонали 1, вне диагонали 100; оптимальная стоимость известна заранее и равна 5. Оба варианта её нашли. Это демонстрационный функциональный тест.\n'
        '6. Для исследовательских результатов использовать финальную серию 7838 запусков; демонстрационные проверки не прибавляются к её числу.\n\n'
        f'Точный оптимум реальной матрицы: {exact}; ГА: {result["cost"]}; средняя приспособленность начальной демонстрационной популяции: {pop.fitness():.12g}.\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
