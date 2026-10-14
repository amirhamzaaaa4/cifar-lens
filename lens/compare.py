"""Compare measured full-data experiments, never historical or smoke scores."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiments', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=Path('evidence/benchmark'))
    args = parser.parse_args()
    results = [json.loads((folder / 'results.json').read_text()) for folder in args.experiments]
    if any(result['smoke_run'] for result in results):
        parser.error('Smoke runs cannot be benchmarked')
    protocols = {(r['seed'], r['training_samples'], r['validation_samples'], r['test_samples'],
                  r['command_config']['epochs'], r['command_config']['batch_size']) for r in results}
    if len(protocols) != 1:
        parser.error('Experiments must use the same seed, split counts, epoch budget and batch size')
    args.output.mkdir(parents=True, exist_ok=True)
    fields = ['architecture', 'accuracy', 'macro_f1', 'ece', 'raw_ece', 'parameters', 'training_seconds']
    rows = [{key: r[key] for key in fields} for r in results]
    with (args.output / 'comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([r['architecture'] for r in results], [r['accuracy'] for r in results], color=['#5CE0C6', '#687FFF'][:len(results)])
    ax.set_ylim(0, 1); ax.set_ylabel('Held-out test accuracy')
    ax.set_title('CIFAR Lens · same evaluation protocol')
    for i, r in enumerate(results):
        ax.text(i, r['accuracy'] + .015, f"{r['accuracy']:.2%}", ha='center')
    fig.tight_layout(); fig.savefig(args.output / 'accuracy.png', dpi=160); plt.close(fig)
    (args.output / 'comparison.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
