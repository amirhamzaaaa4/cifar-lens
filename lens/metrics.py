import numpy as np


def softmax(logits, temperature=1.0):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    values = np.asarray(logits, dtype=np.float64) / temperature
    values -= values.max(axis=-1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=-1, keepdims=True)


def calibration_error(probabilities, labels, bins=15):
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels).reshape(-1)
    if len(labels) == 0 or len(labels) != len(probabilities):
        raise ValueError('Nonempty matching probabilities and labels required')
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    result, rows = 0.0, []
    edges = np.linspace(0, 1, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            accuracy, average = float(correct[mask].mean()), float(confidence[mask].mean())
            result += float(mask.mean()) * abs(accuracy - average)
            rows.append({'confidence': average, 'accuracy': accuracy, 'count': int(mask.sum())})
    return result, rows


def bootstrap_accuracy_ci(correct, samples=2000, seed=73, level=0.95):
    """Percentile bootstrap interval for accuracy over test examples."""
    correct = np.asarray(correct, dtype=np.float64).reshape(-1)
    if len(correct) == 0:
        raise ValueError('At least one prediction required')
    rng = np.random.default_rng(seed)
    if samples < 1 or not 0 < level < 1 or not np.isin(correct, [0, 1]).all():
        raise ValueError('Binary correctness values, positive samples and level in (0, 1) required')
    # Chunk draws to avoid retaining hundreds of MB on a small laptop.
    means = np.concatenate([
        correct[rng.integers(0, len(correct), (min(64, samples - start), len(correct)))].mean(axis=1)
        for start in range(0, samples, 64)
    ])
    tail = (1 - level) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def flip_averaged_logits(predict, batch):
    """Test-time augmentation: mean logits of each image and its mirror image.

    `predict` maps an NHWC float batch to logits.
    """
    batch = np.asarray(batch)
    mirrored = np.ascontiguousarray(batch[:, :, ::-1, :])
    return (np.asarray(predict(batch)) + np.asarray(predict(mirrored))) / 2


def fit_temperature(logits, labels):
    """Select temperature on validation predictions only, never test labels."""
    labels = np.asarray(labels).reshape(-1)
    candidates = np.geomspace(0.5, 5.0, 100)
    losses = [-np.log(softmax(logits, t)[np.arange(len(labels)), labels].clip(1e-12)).mean() for t in candidates]
    return float(candidates[np.argmin(losses)])
