"""Verify a completed final study and package presentation materials."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results/final_offspring_20261028')
    p.add_argument('--archive',required=True)
    args=p.parse_args()
    root=Path(args.output)
    meta=json.loads((root/'metadata.json').read_text())
    audit=json.loads((root/'audit.json').read_text())
    assert meta['status']=='completed' and audit['status']=='passed'
    assert meta['actual_runs']==sum(audit['actual_runs'].values())
    files=sorted(Path('lab1').glob('*.py'))+[Path('requirements.txt')]
    (root/'source_hashes.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
    manifest={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
              for f in root.rglob('*') if f.is_file() and f.name!='manifest.json'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    assert all(hashlib.sha256((root/f).read_bytes()).hexdigest()==expected for f,expected in manifest.items())
    archive=Path(args.archive)
    with ZipFile(archive,'w',compression=ZIP_DEFLATED) as z:
        for file in sorted(root.rglob('*')):
            if file.is_file():z.write(file,file.relative_to(root))
        for file in files:z.write(file,Path('source')/file)
        protocol=Path('docs/final-offspring-protocol-20261028.md')
        z.write(protocol,'METHOD.md')
    with ZipFile(archive) as z:
        assert z.testzip() is None
        assert len([n for n in z.namelist() if n.endswith('.png')])==48
        assert len([n for n in z.namelist() if n.endswith('.svg')])==48
    print('Verified archive:',archive)
    print('Measured runs:',meta['actual_runs'])


if __name__=='__main__':
    main()
