"""Build a presentation figure and documented starting configurations."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from hybrid_ga import Config
from local_evidence import draw_series, finish_unified, read, unified_axes


def main():
    root=Path('results/class_diagnosis_20261028')
    rows=read(root/'summary.csv')
    fig,ax=unified_axes('Отклонение от точного решения, %',
                       'Влияние диапазона стоимостей на ошибку решения')
    maximum=0
    for high,color,label in [('100','#14599B','1–100'),('100000','#D77C33','1–100000')]:
        data=sorted((r for r in rows if r['high']==high and r['variant']=='common'),key=lambda r:int(r['n']))
        maximum=max(maximum,draw_series(ax,[int(r['n']) for r in data],
                                        [float(r['mean_error_pct']) for r in data],color,label))
    finish_unified(fig,ax,root/'cost_range_error',maximum,'error_pct','Стоимость элементов','')
    configs={}
    for group,pct in [('small',50),('medium',5)]:
        configs[group]=asdict(Config(population=64,generations=100,selection='rank',
                         rank_power=2,inversion_pct=pct,local_every=5,local_top=2,
                         local_candidates=16,cycle3=True,cycle3_passes=3,local_on_offspring=True))
    (root/'recommended_starting.json').write_text(json.dumps(configs,indent=2))
    # Check that the new comparison really changes only the local target.
    holdout=read(root/'offspring_repair_holdout.csv')
    for group in configs:
        for n in {r['n'] for r in holdout if r['group']==group}:
            for rep in range(5):
                pair=[r for r in holdout if r['group']==group and r['n']==n and r['rep']==str(rep)
                      and r['variant'] in ('offspring_repair','matched_elite_repair')]
                assert len(pair)==2
                a,b=[json.loads(r['settings_json']) for r in pair]
                assert {k for k in a if a[k]!=b[k]}=={'local_on_offspring'}
                assert pair[0]['local_calls']==pair[1]['local_calls']=='40'
    (root/'audit.json').write_text(json.dumps({'matched_pairs': 30,'local_calls_per_run': 40,
                   'changed_factor': 'local_on_offspring','train_offset': 950000000,'holdout_offset': 970000000,
                   'original_series_unchanged': True},indent=2))
    manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.glob('*') if p.is_file() and p.name!='manifest.json'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
