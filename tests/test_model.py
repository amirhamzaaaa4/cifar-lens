import numpy as np
from lens.model import build_model
from lens.explain import gradcam


def test_both_architectures_train_and_explain():
    for architecture in ('baseline', 'residual', 'resnet20'):
        model = build_model(architecture)
        x = np.random.default_rng(73).random((2, 32, 32, 3), dtype=np.float32)
        assert model(x, training=False).shape == (2, 10)
        loss = model.train_on_batch(x, np.array([0, 1]))
        assert np.isfinite(loss).all()
        heat = gradcam(model, x[:1])
        assert heat.ndim == 2 and np.isfinite(heat).all()
        assert 0 <= heat.min() <= heat.max() <= 1
