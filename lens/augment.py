"""Training-only image augmentation, applied in the tf.data pipeline.

Keeping augmentation out of the saved model means validation, test and app
inference always see the unmodified 32 x 32 image.
"""
import tensorflow as tf

SIZE = 32


def pad_crop_flip(image, padding=4):
    """Standard CIFAR recipe: zero-pad, take a random 32 x 32 crop, random mirror."""
    image = tf.pad(image, [[padding, padding], [padding, padding], [0, 0]])
    image = tf.image.random_crop(image, [SIZE, SIZE, 3])
    return tf.image.random_flip_left_right(image)


def cutout(image, size=8):
    """Zero one random size x size square; the centre may sit near an edge."""
    if size <= 0:
        return image
    cy = tf.random.uniform([], 0, SIZE, dtype=tf.int32)
    cx = tf.random.uniform([], 0, SIZE, dtype=tf.int32)
    rows = tf.range(SIZE)[:, None]
    cols = tf.range(SIZE)[None, :]
    top, left = cy - size // 2, cx - size // 2
    inside = (rows >= top) & (rows < top + size) & (cols >= left) & (cols < left + size)
    return image * tf.cast(~inside, image.dtype)[..., None]
