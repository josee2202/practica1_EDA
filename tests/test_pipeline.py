"""Checks that run without the real dataset (used in CI)."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_notebooks_have_no_absolute_paths():
    # Notebook 04 used os.chdir('/Users/...') to import src/, which broke on any other machine
    for path in sorted((ROOT / 'notebooks').glob('*.ipynb')):
        nb = json.loads(path.read_text(encoding='utf-8'))
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                assert '/Users/' not in source, f'absolute path in {path.name}: {source[:120]!r}'


def test_pipeline_runs_end_to_end_on_synthetic_data(tmp_path):
    workdir = Path(os.environ.get('SMOKE_WORKDIR', tmp_path / 'smoke'))
    rows = os.environ.get('SMOKE_ROWS', '6000')
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / 'run_pipeline.py'), '--synthetic', '--rows', rows,
         '--workdir', str(workdir), '--timeout', '1800', '--html'],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-4000:] + '\n' + result.stderr[-4000:]

    # --html exercises the nbconvert export used to refresh html/
    assert len(list((workdir / 'html').glob('0[1-5]_*.html'))) == 5

    metrics = json.loads((workdir / 'reports' / 'metrics.json').read_text())
    # The synthetic target follows a known rule, so a working pipeline must clearly beat the
    # no-skill baseline; a broken feature/label alignment would land near 0.5.
    assert metrics['roc_auc'] > metrics['baseline_prior']['roc_auc'] + 0.1
    assert metrics['pr_auc'] > metrics['baseline_prior']['pr_auc']
