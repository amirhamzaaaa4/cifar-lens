import numpy as np
import tensorflow as tf
from PIL import Image


def gradcam(model, batch, class_index=None):
    probe = tf.keras.Model(model.input, [model.get_layer('features').output, model.output])
    with tf.GradientTape() as tape:
        features, logits = probe(batch, training=False)
        target = int(tf.argmax(logits[0])) if class_index is None else class_index
        score = logits[:, target]
    gradients = tape.gradient(score, features)
    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heat = tf.nn.relu(tf.reduce_sum(features[0] * weights, axis=-1))
    return tf.math.divide_no_nan(heat, tf.reduce_max(heat)).numpy()


def overlay(image, heat):
    from matplotlib import colormaps
    image = image.convert('RGB').resize((256, 256))
    scaled = np.asarray(Image.fromarray(np.uint8(heat * 255)).resize((256, 256))) / 255.0
    color = np.uint8(colormaps['inferno'](scaled)[..., :3] * 255)
    return Image.blend(image, Image.fromarray(color), 0.45)
