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
    from PIL import ImageOps, Image
    image = ImageOps.exif_transpose(image).convert('RGB')
    return np.asarray(image.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32)[None] / 255.0
