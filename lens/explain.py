import numpy as np
import tensorflow as tf
from PIL import Image


def build_probe(model):
    """Model returning the Grad-CAM feature map and logits. Build once, reuse."""
    return tf.keras.Model(model.input, [model.get_layer('features').output, model.output])


def gradcam(model, batch, class_index=None, probe=None):
    probe = probe or build_probe(model)
    with tf.GradientTape() as tape:
        features, logits = probe(batch, training=False)
        target = int(tf.argmax(logits[0])) if class_index is None else class_index
        score = logits[:, target]
    gradients = tape.gradient(score, features)
    weights = tf.reduce_mean(tf.cast(gradients, tf.float32), axis=(0, 1, 2))
    heat = tf.nn.relu(tf.reduce_sum(tf.cast(features[0], tf.float32) * weights, axis=-1))
    return tf.math.divide_no_nan(heat, tf.reduce_max(heat)).numpy()


def overlay(image, heat):
    from matplotlib import colormaps
    image = image.convert('RGB').resize((256, 256))
    scaled = np.asarray(Image.fromarray(np.uint8(heat * 255)).resize((256, 256), Image.Resampling.BILINEAR)) / 255.0
    color = np.uint8(colormaps['inferno'](scaled)[..., :3] * 255)
    return Image.blend(image, Image.fromarray(color), 0.45)
