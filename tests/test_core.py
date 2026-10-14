import numpy as np
import pytest
from PIL import Image
from lens.data import stratified_split, prepare_image
from lens.metrics import (softmax, calibration_error, fit_temperature,
                          bootstrap_accuracy_ci, flip_averaged_logits)


def test_split_is_disjoint_balanced_and_repeatable():
    labels = np.repeat(np.arange(10), 100)
    train, val = stratified_split(labels)
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(range(1000))
    assert np.array_equal(np.bincount(labels[val]), np.full(10, 10))
    assert np.array_equal(val, stratified_split(labels)[1])


def test_rgb_contract():
    batch = prepare_image(Image.new('L', (90, 10), 255))
    assert batch.shape == (1, 32, 32, 3) and batch.dtype == np.float32
    assert np.all(batch == 1)


def test_softmax_is_stable():
    assert np.allclose(softmax([[10000, 10000]]), [[.5, .5]])
    with pytest.raises(ValueError):
        softmax([[1, 2]], 0)


def test_calibration_counts_confidence_one():
    error, rows = calibration_error(np.array([[1., 0.], [0., 1.]]), [0, 0])
    assert error == .5 and sum(row['count'] for row in rows) == 2


def test_temperature_reduces_validation_nll():
    logits = np.array([[8, 0], [8, 0], [0, 8], [0, 8]])
    labels = np.array([0, 1, 1, 1])
    temperature = fit_temperature(logits, labels)
    def nll(t):
        return -np.log(softmax(logits, t)[np.arange(4), labels]).mean()
    assert nll(temperature) < nll(1)


def test_wide_image_is_centre_cropped_not_squashed():
    image = Image.new('RGB', (300, 100), (255, 0, 0))
    image.paste((0, 0, 255), (100, 0, 200, 100))  # blue square in the centre
    batch = prepare_image(image)
    # A plain resize would leave only the middle third blue (mean ~0.33).
    assert batch[0, ..., 2].mean() > 0.95 and batch[0, 4:-4, 4:-4, 0].max() < 0.05


def test_bootstrap_interval_brackets_accuracy():
    correct = np.r_[np.ones(900), np.zeros(100)]
    low, high = bootstrap_accuracy_ci(correct)
    assert low < 0.9 < high and high - low < 0.05
    assert bootstrap_accuracy_ci(correct) == (low, high)


def test_flip_tta_averages_image_and_mirror():
    batch = np.zeros((1, 2, 2, 1), dtype=np.float32)
    batch[0, 0, 0, 0] = 1
    seen = []

    def predict(images):
        seen.append(images.copy())
        return images.reshape(len(images), -1)
    logits = flip_averaged_logits(predict, batch)
    assert seen[1][0, 0, 1, 0] == 1
    assert np.allclose(logits, [[.5, .5, 0, 0]])
