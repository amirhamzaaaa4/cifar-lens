"""Integration checks against real saved weights when local artifacts exist."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image, UnidentifiedImageError
from lens.data import prepare_image
from lens.metrics import softmax


def test_saved_model_predicts_uploaded_image_and_explains():
    folder = Path(__file__).parents[1] / 'artifacts' / 'smoke'
    if not (folder / 'model.keras').exists():
        pytest.skip('No local saved model; model integration is covered separately')
    import tensorflow as tf
    from lens.explain import gradcam, overlay
    results = json.loads((folder / 'results.json').read_text())
    path = folder / 'model.keras'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == results['model_sha256']
    model = tf.keras.models.load_model(path)
    errors = np.load(folder / 'errors.npz')
    source = Image.fromarray(errors['images'][0])
    stream = io.BytesIO()
    source.save(stream, format='PNG'); stream.seek(0)
    batch = prepare_image(Image.open(stream))
    scores = softmax(model(batch, training=False).numpy(), results['temperature'])
    assert scores.shape == (1, 10)
    assert np.isfinite(scores).all() and np.allclose(scores.sum(axis=1), 1)
    heat = gradcam(model, batch)
    assert overlay(source, heat).size == (256, 256)
    with pytest.raises(UnidentifiedImageError):
        Image.open(io.BytesIO(b'this is not an image'))
