"""Audit and publish ONE complete final version of the offspring GA study."""
import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import local_evidence as evidence
import local_research as research
from hybrid_ga import Config


def extras(root,data):
    colors=('#14599B','#17967F','#D77C33')
    for metric,ylabel in [('error_pct','Отклонение от точного решения, %'),('seconds','Время работы ГА, с')]:
        fig,ax=evidence.unified_axes(ylabel,'Общие и подобранные настройки — '+('точность решения' if metric=='error_pct' else 'время работы'))
        maximum=0
        for i,(variant,label) in enumerate([('common','Одинаковые настройки'),('single','Настройки для каждого класса')]):
            ns,ys=evidence.measured_series([r for r in data['survey'] if r['variant']==variant],metric)
            maximum=max(maximum,evidence.draw_series(ax,ns,ys,colors[i],label))
        evidence.finish_unified(fig,ax,root/'plots'/f'common_settings_{metric}',maximum,metric,'Настройки','')
    fig,ax=evidence.unified_axes('Отклонение от точного решения, %','Выбор особей для локального поиска — контрольная проверка')
    maximum=0
    for i,(variant,label) in enumerate([('single','Подобранный вариант'),('without_offspring','Поиск по лучшим особям'),('parallel','Островной вариант')]):
        grouped=defaultdict(list)
        for row in data['holdout']:
            if row['variant']==variant:grouped[int(row['n'])].append(float(row['error_pct']))
        ns=sorted(grouped);ys=[mean(grouped[n]) for n in ns]
        maximum=max(maximum,evidence.draw_series(ax,ns,ys,colors[i],label))
    evidence.finish_unified(fig,ax,root/'plots'/'holdout_offspring',maximum,'error_pct','Стратегия поиска','')


def effects(root,data):
    paired=defaultdict(list)
    for row in data['survey']:
        if row['parameter']:
            paired[(row['class_name'],row['n'],row['repeat'],row['parameter'])].append(row)
    rows=[]
    for (group,n,rep,parameter),trials in paired.items():
        context=json.loads(trials[0]['context_json'])
        baseline=next(r for r in trials if r['value']==str(context[parameter]))
        for trial in trials:
            rows.append({'class_name':group,'n':n,'repeat':rep,'parameter':parameter,
                         'value':trial['value'],'error_delta_pp':float(trial['error_pct'])-float(baseline['error_pct']),
                         'seconds_delta':float(trial['seconds'])-float(baseline['seconds'])})
    evidence.save(root/'paired_parameter_effects.csv',rows)


def documents(root,data,selected,stats,audit):
    by_key={(r['class_name'],r['variant']):r for r in stats}
    label={'small':'Малые','medium':'Средние','large':'Большие'}
    summary='| Класс | Обычный ГА, ошибка | Островной ГА, ошибка | Без поиска по потомкам | Обычный / островной, мс |\n|---|---:|---:|---:|---:|\n'
    for group in research.CLASS_GRID:
        a=by_key[group,'single'];b=by_key[group,'parallel'];c=by_key[group,'without_offspring']
        summary+=f"| {label[group]} | {a['mean_error_pct']:.2f}% | {b['mean_error_pct']:.2f}% | {c['mean_error_pct']:.2f}% | {1000*a['mean_seconds']:.1f} / {1000*b['mean_seconds']:.1f} |\n"
    meta=json.loads((root/'metadata.json').read_text())
    counts=audit['actual_runs'];total=sum(counts.values())
    text=f'''# Итоговый комплект для презентации — 28.10.2026

Полная серия обновлённого алгоритма завершена. **{total} реальных запусков**,
{meta['wall_seconds']/60:.2f} минут локального расчёта. Все таблицы и графики
в этом каталоге относятся к одной версии алгоритма и одной методике.

## Независимая контрольная проверка

{summary}
Ошибка: 100 × (стоимость ГА − точный оптимум SciPy) / точный оптимум.
«Без поиска по потомкам» отличается от обычного варианта только выключением
`local_on_offspring`; если TRAIN сам выключил этот флаг, варианты совпадают.
Одинаковое число локальных вызовов не означает одинаковое фактическое время.

## Что можно утверждать на защите

- Исследованы инверсия, RSS/ранговый RSS, элитизм, парные и тройные обмены,
  локальный поиск по потомкам и настоящие параллельные острова.
- Пара обычный/островной имеет одинаковые N, G, параметры локального поиска
  и фактическую глобальную квоту локальных вызовов. Различаются только острова
  и миграция; измеренное время включает передачу данных и миграцию.
- Меньшая ошибка больших матриц зависит от диапазона стоимостей 1–100 и
  количества равноценных дешёвых назначений. Это не доказательство того,
  что большие задачи вообще проще. Отдельные графики `common_settings_*`
  фиксируют одинаковые настройки для всех размеров и отделяют этот эффект
  от настройки классов.
- Слабое влияние некоторых параметров — результат измерений. Близкие линии
  не доказывают отсутствие эффекта; в таблице парных разностей сохранены
  изменения ошибки и времени на одних матрицах.
- Лучшие настройки — лучшие в ограниченном последовательном TRAIN-поиске,
  а не глобальные оптимумы и не гарантия качества на другом семействе данных.

## Методика

TRAIN: {counts['train']} запусков, по 3 размера и 2 независимые матрицы каждого
класса. Один координатный проход: в каждом сравнении меняется одно поле.
Текущий уровень всегда входит в сравнение; на равенстве сохраняется текущий.
После обучения настройки заморожены в `selected.json`.

SURVEY: {counts['survey']} запусков, 46 размеров × **2 независимые матрицы**.
Малые 10–90, шаг 10; средние 100–475, шаг 25; большие 500–950, шаг 50,
1000–2000, шаг 100. Строгий OFAT по {len(research.LEVELS)} параметру, плюс
парные обычный/островной и общий для всех классов базовый вариант.
Точки кривых — средние измеренных запусков на данном размере.
Условные параметры исследованы в активном фиксированном контексте:
степень рангов при ранговом RSS, управляющие поля адаптации при включённой
адаптации, параметры миграции при нескольких островах и ненулевой доле
мигрантов для исследования интервала. Полный контекст сохранён в каждой строке.

HOLDOUT: {counts['holdout']} запусков, 12 размеров × **5 независимых матриц**,
6 вариантов. Они не участвовали в выборе параметров. Диагональная проверка:
{counts['diagonal']} запусков, во всех найден точный оптимум n.
Seed всех этапов непересекаются и отличаются от предыдущих исследований.

Графики: Montserrat, все три класса на одном графике, подписи для презентации.
Плавное соединение PCHIP только оформляет измеренные средние и не создаёт
наблюдений. Ось X состоит из трёх непрерывных участков разного масштаба;
на части графиков ошибка показана с нелинейной шкалой. Эти пояснения вынесены
сюда и не загромождают рисунки. Базовые настройки OFAT различаются между
классами; переход между ними не является отдельным наблюдением.

Времена получены с прогретыми функциями и общим переиспользуемым пулом
из четырёх процессов. Создание пула и SciPy исключены, сериализация матриц
и миграция включены. Фоновая нагрузка ОС не контролировалась.

## Материалы

[Лучшие параметры](best_parameters.csv) · [Контрольные итоги](holdout_summary.csv)
· [Чувствительность](ofat_sensitivity.csv) · [Парные эффекты](paired_parameter_effects.csv)
· [Аудит](audit.json) · [Параметры и seed](metadata.json).

## Графики сравнения

![Контроль](plots/holdout_offspring.png)

[Контроль SVG](plots/holdout_offspring.svg) · [Качество обычного/островного](plots/paired_survey_error_pct.png)
· [Время](plots/paired_survey_seconds.png) · [Общие настройки](plots/common_settings_error_pct.png).

'''
    for parameter in research.LEVELS:
        name=evidence.LABELS[parameter]
        text+=f'## {name}\n\n![{name}](plots/ofat_{parameter}_error_pct.png)\n\n[PNG](plots/ofat_{parameter}_error_pct.png) · [SVG](plots/ofat_{parameter}_error_pct.svg) · [Время PNG](plots/ofat_{parameter}_seconds.png) · [Время SVG](plots/ofat_{parameter}_seconds.svg).\n\n'
    common_note=(f"На независимом контроле общая конфигурация дала "
                 f"{by_key['small','common']['mean_error_pct']:.2f}% на малых и "
                 f"{by_key['medium','common']['mean_error_pct']:.2f}% на средних, "
                 "то есть оказалась точнее подобранных на TRAIN вариантов. "
                 "Это показывает ограниченность координатного поиска и его обучающей выборки. "
                 "Параметры по HOLDOUT не заменялись; этот результат сохранён как проверка переносимости.")
    text+='\n## Переносимость выбранных настроек\n\n'+common_note+'\n'
    (root/'README.md').write_text(text.rstrip()+'\n')
    configs=[]
    for group in research.TRAIN:
        for variant in ('single','parallel'):
            configs.append({'class':group,'variant':variant,**asdict(Config(**selected[group][variant]))})
    evidence.save(root/'presentation_parameters.csv',configs)
    parameter_text='# Параметры для презентации\n\n| Параметр | Малые | Средние | Большие |\n|---|---:|---:|---:|\n'
    fields=[('Общая популяция','population'),('Поколения','generations'),
            ('Вероятность мутации','mutation_prob'),('Длина инверсии, %','inversion_pct'),
            ('Доля элиты','elitism'),('Отбор','selection'),('Степень рангов','rank_power'),
            ('Период локального поиска','local_every'),('Общая квота за событие','local_top'),
            ('Кандидаты K','local_candidates'),('Проходы парного поиска','local_passes'),
            ('Проходы тройного поиска','cycle3_passes'),('Поиск по потомкам','local_on_offspring')]
    for title,field in fields:
        values=[str(selected[g]['single'][field]) for g in research.TRAIN]
        values=[{'True':'Да','False':'Нет','rank':'Ранговый RSS','raw':'RSS'}.get(v,v) for v in values]
        parameter_text+='| '+title+' | '+' | '.join(values)+' |\n'
    for title,values in [
        ('Лучшее число островов на TRAIN',[selected[g]['recommended_islands'] for g in research.TRAIN]),
        ('Острова в параллельном сравнении',[selected[g]['parallel']['islands'] for g in research.TRAIN]),
        ('Интервал миграции',[selected[g]['parallel']['migration_interval'] if selected[g]['parallel']['migration_pct'] else '—' for g in research.TRAIN]),
        ('Доля мигрантов',[selected[g]['parallel']['migration_pct'] for g in research.TRAIN])]:
        parameter_text+='| '+title+' | '+' | '.join(map(str,values))+' |\n'
    (root/'PARAMETERS.md').write_text(parameter_text)
    (root/'CONCLUSIONS.md').write_text('# Итоги для презентации\n\n'+summary+
        '\nНастройки выбраны на независимой обучающей выборке и затем заморожены. '
        'В каждой контрольной точке пять независимых матриц. '
        'Сравнение обычного и островного алгоритмов использует равные общие популяции, поколения и квоты локального поиска. '
        'Меньшая относительная ошибка больших задач связана с большим количеством дешёвых назначений в матрицах 1–100. '
        'Островная модель не гарантирует ускорения; вывод о скорости нужно делать по измеренному времени. '
        'Лучшие параметры означают лучшие среди проверенных, а не глобальный оптимум.\n\n'+common_note+'\n')



def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/final_offspring_20261028')
    args=p.parse_args();root=Path(args.output)
    data,selected,audit=evidence.audit(root)
    stats=evidence.tables(root,data,selected)
    evidence.plots(root,data)
    extras(root,data);effects(root,data);documents(root,data,selected,stats,audit)
    manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.rglob('*') if p.is_file() and p.name!='manifest.json'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(audit,indent=2))
    for row in stats:print(row)


if __name__=='__main__':
    main()
