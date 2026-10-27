#!/usr/bin/env python
"""Run the project notebooks (01 -> 05) top to bottom, each in a fresh kernel.

Real data (default): executes the notebooks in place, so their saved outputs are
refreshed, and checks the artifacts they produce.

    python scripts/run_pipeline.py            # full run on data/raw_data/application_data.csv
    python scripts/run_pipeline.py --html     # also re-export html/

Synthetic smoke run: copies notebooks/ and src/ into a scratch workspace, writes a
small synthetic dataset with the same schema and runs the same notebooks there.
It checks that the code runs end to end; its metrics say nothing about real data.

    python scripts/run_pipeline.py --synthetic --rows 6000 --workdir build/smoke
"""
import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
STAGES = ['01', '02', '03', '04', '05']
RAW_DATA = Path('data/raw_data/application_data.csv')


def find_notebook(nb_dir, prefix):
    # Glob instead of literal names: '01_Exploración_general' has a non-ASCII character
    matches = sorted(nb_dir.glob(f'{prefix}_*.ipynb'))
    if len(matches) != 1:
        sys.exit(f'Expected exactly one notebook {prefix}_*.ipynb in {nb_dir}, found {len(matches)}')
    return matches[0]


def execute(path, timeout):
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name='python3',
                            resources={'metadata': {'path': str(path.parent)}})
    client.execute()
    nbformat.write(nb, path)
    return nb


def export_html(nb, path, html_dir):
    from nbconvert import HTMLExporter

    html_dir.mkdir(parents=True, exist_ok=True)
    body, _ = HTMLExporter().from_notebook_node(nb)
    (html_dir / f'{path.stem}.html').write_text(body, encoding='utf-8')


def prepare_synthetic_workspace(workdir, rows, seed):
    sys.path.insert(0, str(ROOT / 'scripts'))
    from make_synthetic_data import make_synthetic_application_data

    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    (workdir / 'notebooks').mkdir()
    for prefix in STAGES:
        shutil.copy2(find_notebook(ROOT / 'notebooks', prefix), workdir / 'notebooks')
    shutil.copytree(ROOT / 'src', workdir / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    for sub in ['data/raw_data', 'data/processed_data', 'data/variables', 'modelos', 'reports/figures']:
        (workdir / sub).mkdir(parents=True, exist_ok=True)
    df = make_synthetic_application_data(rows, seed=seed)
    df.to_csv(workdir / RAW_DATA, index=False)
    print(f'Synthetic dataset: {df.shape[0]} rows x {df.shape[1]} columns, '
          f'positive rate {df["TARGET"].mean():.3f} -> {workdir / RAW_DATA}')


def validate_outputs(root):
    """Check the artifacts written by the full 01 -> 05 run; returns the test metrics."""
    import joblib
    import numpy as np
    import pandas as pd

    processed = root / 'data/processed_data'
    for name in ['train_data_processed.csv', 'test_data_processed.csv',
                 'df_train_scaled.csv', 'df_test_scaled.csv']:
        assert (processed / name).is_file(), f'missing {processed / name}'

    train = pd.read_csv(processed / 'df_train_scaled.csv')
    test = pd.read_csv(processed / 'df_test_scaled.csv')
    assert list(train.columns) == list(test.columns), 'train/test feature columns differ'
    assert train.columns[-1] == 'TARGET', 'TARGET must be the last column'
    assert not train.isnull().any().any() and not test.isnull().any().any(), 'NaN in model inputs'

    model = joblib.load(root / 'modelos/regresionlog.joblib')
    x_test = test.drop(columns='TARGET')
    assert model.n_features_in_ == x_test.shape[1], 'model and data have different feature counts'
    proba = model.predict_proba(x_test)[:, 1]
    assert np.all((proba >= 0) & (proba <= 1))

    threshold = json.loads((root / 'modelos/umbral_f2.json').read_text())['umbral_f2_validacion']
    assert 0 < threshold < 1, f'threshold out of range: {threshold}'

    metrics = json.loads((root / 'reports/metrics.json').read_text())
    for key in ['roc_auc', 'pr_auc', 'threshold_0.5', 'threshold_f2_validation', 'baseline_prior']:
        assert key in metrics, f'reports/metrics.json lacks {key}'
    assert 0 <= metrics['roc_auc'] <= 1 and 0 <= metrics['pr_auc'] <= 1
    for fig in ['roc_curve.png', 'pr_curve.png', 'confusion_matrices.png', 'shap_importance.png']:
        assert (root / 'reports/figures' / fig).is_file(), f'missing reports/figures/{fig}'
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--synthetic', action='store_true', help='smoke run on synthetic data in a scratch workspace')
    parser.add_argument('--rows', type=int, default=6000, help='rows of synthetic data (default: 6000)')
    parser.add_argument('--seed', type=int, default=0, help='seed for the synthetic data (default: 0)')
    parser.add_argument('--workdir', type=Path, help='workspace for --synthetic (default: a new temp dir)')
    parser.add_argument('--start', choices=STAGES, default='01', help='first notebook to run')
    parser.add_argument('--stop', choices=STAGES, default='05', help='last notebook to run')
    parser.add_argument('--html', action='store_true', help='export each executed notebook to html/')
    parser.add_argument('--timeout', type=int, default=-1, help='per-cell timeout in seconds (default: none)')
    args = parser.parse_args()

    if args.synthetic:
        root = (args.workdir or Path(tempfile.mkdtemp(prefix='credit-risk-smoke-'))).resolve()
        prepare_synthetic_workspace(root, args.rows, args.seed)
    else:
        root = ROOT
        if args.start == '01' and not (root / RAW_DATA).is_file():
            sys.exit(f'Raw data not found at {root / RAW_DATA}. See the README section "Data".')

    stages = STAGES[STAGES.index(args.start):STAGES.index(args.stop) + 1]
    for prefix in stages:
        path = find_notebook(root / 'notebooks', prefix)
        started = time.time()
        print(f'[{prefix}] running {path.name} ...', flush=True)
        nb = execute(path, args.timeout)
        print(f'[{prefix}] done in {time.time() - started:.0f}s', flush=True)
        if args.html:
            export_html(nb, path, root / 'html')

    if stages[-1] == '05':
        metrics = validate_outputs(root)
        print('Artifacts OK. Test metrics (threshold 0.5): '
              + json.dumps(metrics['threshold_0.5']) + f", ROC AUC {metrics['roc_auc']:.3f}, PR AUC {metrics['pr_auc']:.3f}")
    print(f'Workspace: {root}')


if __name__ == '__main__':
    main()
