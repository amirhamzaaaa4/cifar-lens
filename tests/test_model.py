import numpy as np
import pytest
import tensorflow as tf
from lens.augment import cutout, pad_crop_flip
from lens.explain import build_probe, gradcam
from lens.model import ARCHITECTURES, build_model


@pytest.mark.parametrize('architecture', ARCHITECTURES)
def test_architectures_train_and_explain(architecture):
    model = build_model(architecture)
    x = np.random.default_rng(73).random((2, 32, 32, 3), dtype=np.float32)
    assert model(x, training=False).shape == (2, 10)
    loss = model.train_on_batch(x, np.array([0, 1]))
    assert np.isfinite(loss).all()
    heat = gradcam(model, x[:1], probe=build_probe(model))
    assert heat.ndim == 2 and np.isfinite(heat).all()
    assert 0 <= heat.min() <= heat.max() <= 1


def test_resnet20_has_no_inference_time_augmentation():
    """Inference must see the exact input; augmentation belongs in the data pipeline."""
    model = build_model('resnet20')
    assert not any(isinstance(layer, (tf.keras.layers.RandomCrop, tf.keras.layers.RandomFlip,
                                      tf.keras.layers.ZeroPadding2D)) for layer in model.layers)
    x = np.random.default_rng(1).random((1, 32, 32, 3), dtype=np.float32)
    assert np.allclose(model(x, training=False), model(x, training=False))
    assert model.get_layer('features').output.shape[1:3] == (8, 8)


def test_label_smoothing_uses_one_hot_targets():
    model = build_model('resnet20', label_smoothing=0.1)
    x = np.random.default_rng(2).random((2, 32, 32, 3), dtype=np.float32)
    loss = model.train_on_batch(x, tf.one_hot([3, 7], 10))
    assert np.isfinite(loss).all()


def test_pipeline_augmentation_keeps_shape_and_range():
    image = tf.ones((32, 32, 3))
    augmented = pad_crop_flip(image)
    assert augmented.shape == (32, 32, 3)
    assert 0 <= float(tf.reduce_min(augmented)) and float(tf.reduce_max(augmented)) <= 1
    masked = cutout(image, 8)
    erased = 32 * 32 - int(tf.reduce_sum(masked[..., 0]))
    assert 0 < erased <= 64
    assert np.array_equal(cutout(image, 0), image)


def test_mixed_precision_trains_and_exports_float32():
    previous = tf.keras.mixed_precision.global_policy()
    x = np.random.default_rng(3).random((2, 32, 32, 3), dtype=np.float32)
    try:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        model = build_model('resnet20', label_smoothing=0.1)
        assert np.isfinite(model.train_on_batch(x, tf.one_hot([1, 2], 10))).all()
        assert model(x, training=False).dtype == tf.float32
        tf.keras.mixed_precision.set_global_policy('float32')
        exported = build_model('resnet20', label_smoothing=0.1)
        exported.set_weights(model.get_weights())
        assert all(layer.compute_dtype == 'float32' for layer in exported.layers)
        assert np.isfinite(exported(x, training=False)).all()
    finally:
        tf.keras.mixed_precision.set_global_policy(previous)
