"""Independent completeness checks for the teacher's final assignment."""
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from local_evidence import audit
from PIL import Image


def main():
    root=Path('results/final_offspring_20261028')
    out=Path('results/teacher_requirements_audit_20261028');out.mkdir(exist_ok=True)
    data,selected,_base_audit=audit(root)
    metadata=json.loads((root/'metadata.json').read_text())
    assert metadata['status']=='completed' and metadata['stage']=='all'
    counts={k:len(v) for k,v in data.items()}
    assert counts=={'train':1026,'survey':6440,'holdout':360,'diagonal':12}
    assert sum(counts.values())==metadata['actual_runs']==7838
    survey_reps=range(metadata['repeats']['survey'])
    control_reps=range(metadata['repeats']['holdout'])
    variants={'single','parallel','without_cycle3','without_offspring','original_rss_inversion','common'}
    for group,sizes in metadata['holdout_sizes'].items():
        actual=Counter((int(r['n']),int(r['repeat']),r['variant']) for r in data['holdout'] if r['class_name']==group)
        expected=Counter((n,rep,v) for n in sizes for rep in control_reps for v in variants)
        assert actual==expected
    for group,sizes in metadata['survey_sizes'].items():
        actual=Counter((int(r['n']),int(r['repeat']),r['variant']) for r in data['survey'] if r['class_name']==group and r['variant'])
        expected=Counter((n,rep,v) for n in sizes for rep in survey_reps for v in ('single','parallel','common'))
        assert actual==expected
    training=defaultdict(list)
    for row in data['train']:
        training[(row['class_name'],row['phase'],row['parameter'],row['context_json'],row['value'])].append(row)
    for (group,*_),trials in training.items():
        assert Counter((int(r['n']),int(r['repeat'])) for r in trials)==Counter((n,rep) for n in metadata['train_sizes'][group] for rep in (0,1))
    assert len(training)==171
    assert all(r['cost']==r['exact'] for r in data['diagonal'])
    for group,spec in selected.items():
        assert spec['single']['islands']==1
        assert spec['parallel']['islands']>1 and spec['parallel']['migration_pct']>0
        levels={int(r['value']) for r in data['survey'] if r['class_name']==group and r['parameter']=='islands'}
        assert levels=={1,2,4,8}
    expected_plots={f'ofat_{p}_{metric}' for p in metadata['levels'] for metric in ('error_pct','seconds')}
    expected_plots|={'paired_survey_error_pct','paired_survey_seconds','holdout_quality','holdout_offspring','common_settings_error_pct','common_settings_seconds'}
    assert len(expected_plots)==48
    for extension in ('png','svg'):
        assert {p.stem for p in (root/'plots').glob('*.'+extension)}==expected_plots
    for name in expected_plots:
        with Image.open(root/'plots'/f'{name}.png') as im:
            im.verify()
        text=(root/'plots'/f'{name}.svg').read_text()
        assert len(re.findall(r'id="axes_\d+"',text))==1 and 'Montserrat' in text
        assert 'PCHIP' not in text and 'три участка' not in text
    manifest=json.loads((root/'manifest.json').read_text())
    for file,digest in manifest.items():
        assert hashlib.sha256((root/file).read_bytes()).hexdigest()==digest, file
    remote=subprocess.check_output(['git','rev-parse','origin/main'],text=True).strip()
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    assert remote==head
    tree=subprocess.check_output(['git','ls-tree','-r','--name-only','origin/main',str(root)],text=True).splitlines()
    local=[str(p) for p in root.rglob('*') if p.is_file()]
    assert set(local)==set(tree)
    for file in tree:
        blob=subprocess.check_output(['git','show',f'origin/main:{file}'])
        assert hashlib.sha256(blob).digest()==hashlib.sha256(Path(file).read_bytes()).digest(),file
    archive=Path('/Users/rvetlyakov/Documents/IAD-final-presentation-materials-2026-10-28.zip')
    with ZipFile(archive) as z:
        assert z.testzip() is None
        for file,digest in manifest.items():
            assert hashlib.sha256(z.read(file)).hexdigest()==digest,file
    results={'status':'passed','audited_research_commit':head,'counts':counts,
             'training_candidate_evaluations':len(training),'parameters':len(metadata['levels']),
             'sizes':sum(map(len,metadata['survey_sizes'].values())),
             'holdout_variants_complete':sorted(variants),'graphs_png':48,'graphs_svg':48,
             'all_research_files_identical_to_github_main':len(tree),
             'manifest_files_verified':len(manifest),'zip_matches_research_manifest':True,
             'all_diagonal_cases_exact':True,'source_spreadsheet':'ИАД, группа 2, первая версия; текущая версия проверена по коду'}
    (out/'completeness.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
    print(json.dumps(results,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
