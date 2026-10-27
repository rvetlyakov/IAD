"""Retain inactive observations as archive; measure active migration interval.

No retuning: frozen winners and HOLDOUT remain untouched.
"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from hybrid_ga import Config
from local_evidence import read, save
from local_research import Study, survey_context, warm_worker


def main():
    root=Path('results/local_work_20261028')
    rows=read(root/'survey.csv')
    inactive=[r for r in rows if r['parameter']=='migration_interval'
              and not json.loads(r['settings_json'])['migration_pct']]
    if not inactive:
        raise SystemExit('No inactive interval observations to replace')
    selected=json.loads((root/'selected.json').read_text())
    save(root/'archived_inactive_interval.csv',inactive)
    new_root=root/'active_context_recheck'
    if (new_root/'survey.csv').exists():
        raise SystemExit('Recheck already exists')
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(warm_worker,range(8)))
        study=Study(new_root,pool,4)
        for row in inactive:
            group=row['class_name']
            base=Config(**selected[group]['single'])
            parallel=Config(**selected[group]['parallel'])
            context=survey_context(base,parallel,'migration_interval')
            from dataclasses import replace
            study.measure('survey',group,int(row['n']),int(row['repeat']),
                          replace(context,migration_interval=int(row['value'])),
                          'migration_interval',context,phase='active_context_recheck')
    # Replace by keys only after every required new observation was saved.
    replacements=read(new_root/'survey.csv')
    assert len(replacements)==len(inactive)
    fixed=[r for r in rows if r not in inactive]+replacements
    save(root/'survey.csv',fixed)
    metadata=json.loads((root/'metadata.json').read_text())
    metadata['active_interval_recheck_runs']=len(replacements)
    metadata['retained_primary_runs']=metadata['actual_new_runs']
    metadata['total_research_calls_including_archived']=metadata['actual_new_runs']+len(replacements)
    metadata['recheck_reason']='Small TRAIN winner has migration_pct=0. Interval OFAT needs fixed active context migration_pct=0.1. Original 27 inactive runs retained in archive, not plotted. No TRAIN/HOLDOUT changes.'
    (root/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print('Replaced',len(replacements),'inactive rows with new actual active-context measurements')


if __name__=='__main__':
    main()
