import pytest
from lens.compare import summarize


def record(architecture='resnet20', seed=1, accuracy=0.9, smoke=False, epochs=180, **recipe):
    return {'architecture': architecture, 'seed': seed, 'smoke_run': smoke, 'accuracy': accuracy,
            'macro_f1': accuracy, 'ece': 0.02, 'training_seconds': 100.0, 'parameters': 10,
            'training_samples': 45000, 'validation_samples': 5000, 'test_samples': 10000,
            'command_config': {'epochs': epochs}, 'recipe': recipe}


def test_seeds_are_grouped_and_budgets_may_differ():
    rows = summarize([record(seed=1, accuracy=.91), record(seed=2, accuracy=.93),
                      record('baseline', epochs=50, accuracy=.8)])
    assert len(rows) == 2
    assert rows[0]['runs'] == 2 and rows[0]['accuracy'] == pytest.approx(.92)
    assert rows[0]['accuracy_std'] > 0 and rows[1]['accuracy_std'] == 0


def test_recipes_are_separate_rows():
    rows = summarize([record(), record(label_smoothing=0.1, tta=True)])
    assert [r['recipe'] for r in rows] == ['resnet20 · 180ep', 'resnet20 · 180ep · ls0.1 · tta']


def test_smoke_and_duplicate_seed_rejected():
    with pytest.raises(ValueError):
        summarize([record(smoke=True)])
    with pytest.raises(ValueError):
        summarize([record(seed=1), record(seed=1)])
