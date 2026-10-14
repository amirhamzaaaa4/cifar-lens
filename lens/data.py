import numpy as np

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def stratified_split(labels, fraction=0.1, seed=73):
    """Partition only official training indices; retain each class in validation."""
    if not 0 < fraction < 1:
        raise ValueError('fraction must be between zero and one')
    labels = np.asarray(labels).reshape(-1)
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        count = max(1, round(len(indices) * fraction))
        if count >= len(indices):
            raise ValueError('Each class needs at least two samples')
        validation.extend(indices[:count])
        train.extend(indices[count:])
    return rng.permutation(train), rng.permutation(validation)


def prepare_image(image):
    """Upright RGB, centre-cropped to a square, then resized to 32 x 32.

    Cropping first keeps object proportions; a plain resize would squash wide
    or tall photos.
    """
    from PIL import ImageOps, Image
    image = ImageOps.exif_transpose(image).convert('RGB')
    image = ImageOps.fit(image, (32, 32), Image.Resampling.BICUBIC, centering=(0.5, 0.5))
    return np.asarray(image, dtype=np.float32)[None] / 255.0
