import numpy as np
import pytest
from PIL import Image
from lens.data import stratified_split, prepare_image
from lens.metrics import softmax, calibration_error, fit_temperature


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


def test_wide_image_preserves_square_center():
    image = np.zeros((32, 96, 3), dtype=np.uint8)
    image[:, 32:64, 1] = 255
    batch = prepare_image(Image.fromarray(image))[0]
    assert np.all(batch[..., 1] == 1)
    assert np.all(batch[..., [0, 2]] == 0)


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
