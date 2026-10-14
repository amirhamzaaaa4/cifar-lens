import tensorflow as tf


def build_model(architecture='residual', seed=73):
    """Functional graph with logits and a named feature map for Grad-CAM."""
    tf.keras.utils.set_random_seed(seed)
    L = tf.keras.layers
    inputs = L.Input((32, 32, 3), name='image')
    x = inputs
    if architecture == 'baseline':
        for filters in (32, 64):
            x = L.Conv2D(filters, 3, padding='same', activation='relu')(x)
            x = L.MaxPooling2D()(x)
        x = L.Conv2D(64, 3, padding='same', activation='relu', name='features')(x)
    elif architecture == 'residual':
        x = L.RandomFlip('horizontal', seed=seed)(x)
        x = L.RandomTranslation(0.1, 0.1, seed=seed + 1)(x)
        for stage, filters in enumerate((32, 64, 128)):
            shortcut = L.Conv2D(filters, 1, padding='same', use_bias=False)(x)
            for _ in range(2):
                x = L.Conv2D(filters, 3, padding='same', use_bias=False,
                             kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
                x = L.BatchNormalization()(x)
                x = L.Activation('relu')(x)
            x = L.Add()([x, shortcut])
            x = L.Activation('relu', name='features' if stage == 2 else f'stage_{stage}')(x)
            if stage < 2:
                x = L.MaxPooling2D()(x)
                x = L.Dropout(0.2)(x)
    else:
        raise ValueError('Unknown architecture')
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.3)(x)
    logits = L.Dense(10, name='logits')(x)
    model = tf.keras.Model(inputs, logits, name=f'cifar_{architecture}')
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model
