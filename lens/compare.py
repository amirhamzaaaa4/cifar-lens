"""Compare measured full-data experiments, never historical or smoke scores.

Runs that share an architecture and recipe but use different seeds are grouped
and reported as mean ± standard deviation.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def recipe_label(result):
    recipe = result.get('recipe', {})
    parts = [result['architecture'], f"{result['command_config']['epochs']}ep"]
    if recipe.get('label_smoothing'):
        parts.append(f"ls{recipe['label_smoothing']}")
    if recipe.get('cutout'):
        parts.append(f"cutout{recipe['cutout']}")
    if recipe.get('tta'):
        parts.append('tta')
    if 'batch_size' in result['command_config']:
        parts.append(f"batch{result['command_config']['batch_size']}")
    if recipe.get('mixed_precision'):
        parts.append('mixed')
    if recipe.get('deterministic'):
        parts.append('deterministic')
    return ' · '.join(parts)


def summarize(results):
    """Group runs by recipe; return one row per recipe in input order."""
    if any(result['smoke_run'] for result in results):
        raise ValueError('Smoke runs cannot be benchmarked')
    splits = {(r['training_samples'], r['validation_samples'], r['test_samples']) for r in results}
    if len(splits) != 1:
        raise ValueError('Experiments must use the same training/validation/test split sizes')
    groups = {}
    for result in results:
        groups.setdefault(recipe_label(result), []).append(result)
    rows = []
    for label, runs in groups.items():
        seeds = [r['seed'] for r in runs]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f'Duplicate seed within recipe {label}')
        row = {'recipe': label, 'runs': len(runs), 'seeds': ' '.join(map(str, seeds)),
               'parameters': runs[0]['parameters'],
               'epoch_budget': runs[0]['command_config']['epochs'],
               'batch_size': runs[0]['command_config'].get('batch_size'),
               'epochs_completed': ' '.join(str(r.get('epochs_trained', len(r.get('history', {}).get('loss', [])))) for r in runs)}
        for key in ('accuracy', 'macro_f1', 'ece', 'training_seconds'):
            values = np.array([r[key] for r in runs], dtype=float)
            row[key] = float(values.mean())
            row[f'{key}_std'] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('experiments', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=Path('evidence/benchmark'))
    args = parser.parse_args(argv)
    results = [json.loads((folder / 'results.json').read_text()) for folder in args.experiments]
    try:
        rows = summarize(results)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(rows)), 4))
    accuracy = [r['accuracy'] for r in rows]
    ax.bar(range(len(rows)), accuracy, yerr=[r['accuracy_std'] for r in rows], capsize=6, color='#687FFF')
    ax.set_xticks(range(len(rows)), [r['recipe'] for r in rows], rotation=15, ha='right')
    low = max(0, min(accuracy) - 0.1)
    ax.set_ylim(low, 1)
    ax.set_ylabel('Held-out test accuracy')
    ax.set_title('CIFAR Lens · common split sizes, reported budgets, mean ± std')
    for i, r in enumerate(rows):
        ax.text(i, r['accuracy'] + r['accuracy_std'] + .005, f"{r['accuracy']:.2%}", ha='center')
    fig.tight_layout()
    fig.savefig(args.output / 'accuracy.png', dpi=160)
    plt.close(fig)
    (args.output / 'comparison.json').write_text(json.dumps({'summary': rows, 'runs': results}, indent=2),
                                                 encoding='utf-8')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
