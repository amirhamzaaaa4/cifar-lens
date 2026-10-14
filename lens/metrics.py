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


def fit_temperature(logits, labels):
    """Select temperature on validation predictions only, never test labels."""
    labels = np.asarray(labels).reshape(-1)
    candidates = np.geomspace(0.5, 5.0, 100)
    losses = [-np.log(softmax(logits, t)[np.arange(len(labels)), labels].clip(1e-12)).mean() for t in candidates]
    return float(candidates[np.argmin(losses)])
