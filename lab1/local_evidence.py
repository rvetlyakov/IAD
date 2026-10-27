"""Audit completed local observations and produce presentation plots/tables."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from hybrid_ga import Config
from local_research import CLASS_GRID, HOLDOUT, LEVELS, TRAIN, problem, seed_for

CLASS_LABELS = {'small':'Малые · n < 100', 'medium':'Средние · 100–499',
                'large':'Большие · 500–2000'}
LABELS = {
 'local_on_offspring':'Локальный поиск по потомкам',
 'generations':'Число поколений', 'population':'Общая популяция',
 'selection':'RSS: исходный / ранговый', 'rank_power':'Степень рангового RSS',
 'mutation_prob':'Вероятность мутации', 'inversion_pct':'Максимальная длина инверсии, %',
 'elitism':'Доля элиты', 'local_every':'Период локального поиска',
 'local_candidates':'Число кандидатов локального поиска', 'local_passes':'Число проходов парных обменов',
 'local_top':'Число особей для локального поиска',
 'cycle3':'Тройные обмены', 'cycle3_passes':'Число проходов тройных обменов',
 'adaptive':'Адаптивная инверсия', 'adaptive_horizon':'Горизонт адаптации',
 'adaptive_long_prob':'Вероятность дополнительной инверсии',
 'adaptive_long_pct':'Длина дополнительной инверсии, %',
 'islands':'Число островов',
 'migration_interval':'Интервал кольцевой миграции',
 'migration_pct':'Доля мигрантов'}
COLORS = ['#1565c0','#d45b20','#13866d','#9a45aa','#666b74','#c39112']

def value_key(value):
    """CSV levels 2 and 2.0 denote the same numeric hyperparameter."""
    if isinstance(value,bool):
        return str(value)
    try:
        return format(float(value),'.12g')
    except (ValueError,TypeError):
        return str(value)


def read(path):
    with Path(path).open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))

def save(path, rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def audit_row(row):
    cfg=Config(**json.loads(row['settings_json']))
    context=Config(**json.loads(row['context_json']))
    changed={k for k in asdict(cfg) if getattr(cfg,k)!=getattr(context,k)}
    if row['parameter']=='migration_interval':
        assert cfg.migration_pct>0 and cfg.islands>1
    if row['parameter']:
        assert changed <= {row['parameter']}, ('OFAT', changed)
        assert row['value'] == str(getattr(cfg,row['parameter']))
    n,rep=int(row['n']),int(row['repeat'])
    assert int(row['matrix_seed']) == seed_for(row['stage'],n,rep)
    assert int(row['ga_seed']) == int(row['matrix_seed'])+17
    assert int(row['population']) == cfg.population
    assert int(row['generations']) == cfg.generations
    expected_calls=(cfg.generations//cfg.local_every)*cfg.local_top if cfg.local_every else 0
    assert int(row['local_calls']) == expected_calls
    assert row['permutation_valid'] == 'True'
    assert int(row['cost']) >= int(row['exact']) > 0
    error=100*(int(row['cost'])-int(row['exact']))/int(row['exact'])
    assert abs(error-float(row['error_pct'])) < 1e-9
    assert float(row['seconds']) > 0
    if cfg.islands > 1:
        assert json.loads(row['worker_pids'])
        expected_rounds=(cfg.generations-1)//cfg.migration_interval if cfg.migration_pct else 0
        assert int(row['migration_rounds']) == expected_rounds
    return cfg

def audit(root):
    root=Path(root)
    metadata=json.loads((root/'metadata.json').read_text()) if (root/'metadata.json').exists() else {}
    if metadata:
        assert metadata.get('status')=='completed', 'Research is not completed'
    if metadata.get('profile')=='final_offspring':
        from final_research import configure
        configure()
    repeat_counts=metadata.get('repeats',{'survey':1,'holdout':3})
    selected=json.loads((root/'selected.json').read_text())
    if (root/'frozen_before_holdout.json').exists():
        frozen=json.loads((root/'frozen_before_holdout.json').read_text())
        assert hashlib.sha256((root/'selected.json').read_bytes()).hexdigest()==frozen['selected_sha256']
        assert hashlib.sha256((Path(__file__).parent/'hybrid_ga.py').read_bytes()).hexdigest()==frozen['algorithm_sha256']

    data={stage:read(root/f'{stage}.csv') for stage in ('train','survey','holdout','diagonal')}
    seed_sets={}
    for stage, rows in data.items():
        seed_sets[stage]={int(r['matrix_seed']) for r in rows}
        for row in rows:
            audit_row(row)
            _, optimum=problem(stage,int(row['n']),int(row['repeat']))
            assert int(row['exact']) == optimum
    for a, first in seed_sets.items():
        for b, second in seed_sets.items():
            if a!=b:
                assert first.isdisjoint(second), (a,b)
    survey=data['survey']
    for group,sizes in CLASS_GRID.items():
        for parameter in LEVELS:
            for n in sizes:
                rows=[r for r in survey if r['class_name']==group and r['parameter']==parameter and int(r['n'])==n]
                assert rows, (group,parameter,n)
                expected_values={value_key(v) for v in LEVELS[parameter]}
                context=json.loads(rows[0]['context_json'])
                assert expected_values | {value_key(context[parameter])} == {value_key(r['value']) for r in rows}
                assert len({r['context_json'] for r in rows}) == 1
                assert len(rows) == len({(r['value'],r['repeat']) for r in rows})
                for value in {r['value'] for r in rows}:
                    assert {int(r['repeat']) for r in rows if r['value']==value}==set(range(repeat_counts['survey']))
        for stage,repeats,stage_sizes in [('survey',tuple(range(repeat_counts['survey'])),sizes),('holdout',tuple(range(repeat_counts['holdout'])),HOLDOUT[group])]:
            for n in stage_sizes:
                for rep in repeats:
                    pair=[r for r in data[stage] if r['class_name']==group and int(r['n'])==n and int(r['repeat'])==rep and r['variant'] in ('single','parallel')]
                    assert len(pair)==2
                    first, second=pair
                    for field in ('population','generations','local_calls','matrix_seed','ga_seed'):
                        assert first[field]==second[field], (stage,group,n,field)
                    configs=[json.loads(r['settings_json']) for r in pair]
                    assert {k for k in configs[0] if configs[0][k]!=configs[1][k]} <= {'islands','migration_interval','migration_pct'}
                    assert configs[0]==selected[group][first['variant']]
                    assert configs[1]==selected[group][second['variant']]
    for row in data['diagonal']:
        assert int(row['cost'])==int(row['n'])
    for group in TRAIN:
        assert {int(r['n']) for r in data['train'] if r['class_name']==group}==set(TRAIN[group])
    archived=read(root/'archived_inactive_interval.csv') if (root/'archived_inactive_interval.csv').exists() else []
    report={'status':'passed','archived_inactive_runs':len(archived),
            'research_calls_including_archive':sum(len(v) for v in data.values())+len(archived),
            'actual_runs':{s:len(rows) for s,rows in data.items()},
            'unique_matrices':{s:len(v) for s,v in seed_sets.items()},
            'survey_sizes':{s:len(v) for s,v in CLASS_GRID.items()},
            'checks':['strict one-factor contexts','all declared parameter levels and sizes present',
                      'disjoint stage matrix seeds','SciPy exact costs regenerated',
                      'relative error recomputed','equal N/G/global local calls',
                      'identical local passes/candidates and mutation settings in paired comparison',
                      'worker PIDs and migration counts','runtime permutation assertions',
                      'diagonal solutions exact']}
    (root/'audit.json').write_text(json.dumps(report,indent=2))
    return data,selected,report

def export(fig, path):
    # Reject accidentally clipped observations before saving either format.
    for ax in fig.axes:
        left,right=ax.get_xlim()
        bottom,top=ax.get_ylim()
        for line in ax.lines:
            if line.get_marker() in ('_','|') or line.get_transform()!=ax.transData:
                continue
            for x,y in zip(line.get_xdata(),line.get_ydata(),strict=True):
                if np.isfinite(x) and np.isfinite(y):
                    assert left<=x<=right and bottom-1e-9<=y<=top+1e-9, (path,x,y,ax.get_xlim(),ax.get_ylim())
    for extension in ('png','svg'):
        fig.savefig(path.with_suffix('.'+extension),dpi=160,facecolor='white')
    plt.close(fig)

def axes_for(title, ylabel, grid=None):
    grid=CLASS_GRID if grid is None else grid
    fig,axes=plt.subplots(1,3,figsize=(16,5.5),layout='constrained')
    fig.suptitle(title,fontsize=17,weight='bold')
    for ax,(group,sizes) in zip(axes,grid.items(),strict=True):
        ax.set_title(CLASS_LABELS[group],fontsize=12,pad=12)
        ax.set_xlabel('Размер матрицы n',fontsize=11)
        ax.set_ylabel(ylabel,fontsize=11)
        ax.grid(axis='y',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
        span=max(sizes)-min(sizes)
        ax.set_xlim(min(sizes)-.025*span,max(sizes)+.025*span)
    return fig,axes

def smooth_points(x,y):
    """PCHIP is a display connector, never an additional observation."""
    from scipy.interpolate import PchipInterpolator
    x=np.asarray(x,dtype=float)
    y=np.asarray(y,dtype=float)
    # Include every original X exactly, so measured values remain visible.
    xx=np.unique(np.concatenate((np.linspace(x[0],x[-1],max(300,len(x)*24)),x)))
    return xx,PchipInterpolator(x,y)(xx)


def unified_axes(ylabel,title=None):
    from matplotlib import font_manager
    from staged_charts import x_value
    fonts=list((Path.home()/'Library'/'Fonts').glob('Montserrat-*.ttf'))
    for font in fonts:
        font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':'Montserrat' if fonts else 'DejaVu Sans',
                         'font.size':11,'svg.fonttype':'none','axes.unicode_minus':False})
    fig,ax=plt.subplots(figsize=(14,7.5),dpi=165,facecolor='white')
    fig.subplots_adjust(left=.10,right=.984,top=.90,bottom=.215)
    if title:
        fig.suptitle(title,y=.975,fontsize=17,fontweight='medium')
    fills=('#EAF7F2','#EBF2FB','#FCF2F0')
    ink=('#1E8F7A','#316AA0','#B96857')
    for i,(group,color) in enumerate(zip(CLASS_GRID,fills,strict=True)):
        ax.axvspan(i,i+1,color=color,alpha=.93,zorder=0)
        ax.text(i+.5,-.118,CLASS_LABELS[group].lower(),ha='center',va='top',
                fontsize=11,color=ink[i],weight='bold',transform=ax.get_xaxis_transform())
    for edge in (1,2):
        ax.axvline(edge,color='#C7D4DE',lw=.9,zorder=1)
    ticks=(10,30,50,70,90,100,200,300,400,500,750,1000,1250,1500,1750,2000)
    ax.set_xticks([x_value(n) for n in ticks])
    ax.set_xticklabels([str(n) for n in ticks],fontsize=9)
    ax.set_xlim(-.035,3.035)
    ax.set_axisbelow(True)
    ax.grid(axis='y',color='#DDE4ED',lw=.85)
    ax.set_xlabel('Размерность матрицы n × n',labelpad=52,fontsize=12)
    ax.set_ylabel(ylabel,labelpad=12,fontsize=12)
    ax.tick_params(axis='both',length=0,pad=8,colors='#43566B')
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig,ax


def draw_series(ax, ns, ys, color, label, deviations=None):
    from staged_charts import x_value
    x=np.array([x_value(n) for n in ns])
    y=np.asarray(ys,dtype=float)
    xx,yy=smooth_points(x,y)
    ax.plot(xx,yy,lw=2.7,color=color,label=label,zorder=4,solid_capstyle='round')
    ax.scatter(x,y,s=25,color=color,alpha=.32,edgecolors='white',linewidths=.5,zorder=5)
    if deviations is not None:
        ax.errorbar(x,y,yerr=deviations,fmt='none',ecolor=color,alpha=.5,capsize=3,lw=1.2,zorder=3)
    return float(max(y))


def finish_unified(fig,ax,path,maximum,metric,legend_title,footnote):
    from matplotlib import ticker
    if metric=='error_pct' and maximum>250:
        ax.set_yscale('symlog',linthresh=20,base=10)
        ticks=(0,10,20,50,100,300,1000,3000,5000,10000)
        ax.set_yticks([v for v in ticks if v<=maximum*1.2])
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_ylim(0,max(.01,maximum*1.19))
    handles,labels=ax.get_legend_handles_labels()
    ax.legend(handles,labels,title=legend_title,loc='upper left',
              ncol=2 if len(labels)>3 else 1,fontsize=10,title_fontsize=10.5,
              framealpha=.97,facecolor='white',edgecolor='#CDD8E0')
    export(fig,path)


def measured_series(rows,metric):
    grouped=defaultdict(list)
    for row in rows:
        grouped[int(row['n'])].append(float(row[metric]))
    ns=sorted(grouped)
    return ns,[mean(grouped[n]) for n in ns]


def plots(root,data):
    root=Path(root)
    out=root/'plots'
    out.mkdir(exist_ok=True)
    colors=('#14599B','#17967F','#D77C33','#914E94','#BF5862','#6D7481','#758323')
    symbols={'raw':'RSS · 1/C','rank':'Ранговый RSS','True':'Да','False':'Нет'}
    note='Точки — реальные измерения; плавная линия — PCHIP для отображения. Базовые настройки зафиксированы отдельно для каждого класса.'
    for parameter in LEVELS:
        values=list(dict.fromkeys([value_key(v) for v in LEVELS[parameter]]+
                    [value_key(r['value']) for r in data['survey'] if r['parameter']==parameter]))
        for metric,ylabel in [('error_pct','Отклонение от точного решения, %'),('seconds','Время работы ГА, с')]:
            title=LABELS[parameter]+(' — зависимость от размерности' if metric=='error_pct' else ' — время работы')
            fig,ax=unified_axes(ylabel,title)
            maximum=0.
            for i,value in enumerate(values):
                rows=sorted((r for r in data['survey'] if r['parameter']==parameter and value_key(r['value'])==value),key=lambda r:int(r['n']))
                ns,ys=measured_series(rows,metric)
                label='Без поиска' if parameter=='local_every' and value=='0' else symbols.get(value,value)
                maximum=max(maximum,draw_series(ax,ns,ys,colors[i%len(colors)],label))
            finish_unified(fig,ax,out/f'ofat_{parameter}_{metric}',maximum,metric,'Значение параметра',note)
    for metric,ylabel in [('error_pct','Отклонение от точного решения, %'),('seconds','Время работы ГА, с')]:
        fig,ax=unified_axes(ylabel,'Обычный и островной ГА — '+('ошибка решения' if metric=='error_pct' else 'время работы'))
        maximum=0.
        for i,(variant,label) in enumerate([('single','Обычный ГА'),('parallel','Островной ГА')]):
            rows=sorted((r for r in data['survey'] if r['variant']==variant),key=lambda r:int(r['n']))
            ns,ys=measured_series(rows,metric)
            maximum=max(maximum,draw_series(ax,ns,ys,colors[i],label))
        finish_unified(fig,ax,out/f'paired_survey_{metric}',maximum,metric,'Алгоритм',note)
    fig,ax=unified_axes('Отклонение от точного решения, %','Влияние локального поиска на точность решения')
    maximum=0.
    for i,(variant,label) in enumerate([('single','ГА с тройными обменами'),('parallel','Островной ГА с тройными обменами'),('without_cycle3','ГА с парными обменами')]):
        grouped=defaultdict(list)
        for row in data['holdout']:
            if row['variant']==variant:
                grouped[int(row['n'])].append(float(row['error_pct']))
        ns=sorted(grouped)
        ys=[mean(grouped[n]) for n in ns]
        deviations=[stdev(grouped[n]) for n in ns]
        draw_series(ax,ns,ys,colors[i],label,deviations)
        maximum=max(maximum,max(y+d for y,d in zip(ys,deviations,strict=True)))
    finish_unified(fig,ax,out/'holdout_quality',maximum,'error_pct','Независимый контроль',
                   'Точки: среднее ± стандартное отклонение по 3 независимым матрицам. Плавная линия — PCHIP; контроль не использован при подборе.')


def tables(root,data,selected):
    root=Path(root)
    rows=[]
    for group,config in selected.items():
        for variant in ('single','parallel'):
            rows.append({'class_name':group,'variant':variant,'train_recommended_islands':config['recommended_islands'],**config[variant]})
    save(root/'best_parameters.csv',rows)
    stats=[]
    for group in CLASS_GRID:
        for variant in dict.fromkeys(['single','parallel','without_cycle3','original_rss_inversion']+[r['variant'] for r in data['holdout']]):
            rows=[r for r in data['holdout'] if r['class_name']==group and r['variant']==variant]
            stats.append({'class_name':group,'variant':variant,'runs':len(rows),
                          'mean_error_pct':mean(float(r['error_pct']) for r in rows),
                          'max_error_pct':max(float(r['error_pct']) for r in rows),
                          'sd_error_pct':stdev(float(r['error_pct']) for r in rows),
                          'mean_seconds':mean(float(r['seconds']) for r in rows),
                          'exact_fraction':mean(int(r['cost'])==int(r['exact']) for r in rows)})
    save(root/'holdout_summary.csv',stats)
    # Parameter influence is descriptive SURVEY sensitivity, never retuning.
    influence=[]
    for group in CLASS_GRID:
        for parameter in LEVELS:
            values=defaultdict(list)
            for r in data['survey']:
                if r['class_name']==group and r['parameter']==parameter:
                    values[r['value']].append(float(r['error_pct']))
            for value,errors in values.items():
                influence.append({'class_name':group,'parameter':parameter,'value':value,
                                  'observed_runs':len(errors),'mean_survey_error_pct':mean(errors),
                                  'use':'sensitivity only; not selected on survey'})
    save(root/'ofat_sensitivity.csv',influence)
    return stats

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results/local_work_20261028')
    args=p.parse_args()
    root=Path(args.output)
    data,selected,report=audit(root)
    stats=tables(root,data,selected)
    plots(root,data)
    manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(report,indent=2))
    for row in stats:
        print(row)

if __name__=='__main__':
    main()
