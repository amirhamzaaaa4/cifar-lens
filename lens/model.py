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
    elif architecture == 'resnet20':
        # CIFAR-scale 20-layer residual network, trained from scratch.
        x = L.ZeroPadding2D(4)(x)
        x = L.RandomCrop(32, 32, seed=seed)(x)
        x = L.RandomFlip('horizontal', seed=seed + 1)(x)
        x = L.Rescaling(2., offset=-1.)(x)
        def convolution(value, filters, stride=1, size=3):
            return L.Conv2D(filters, size, strides=stride, padding='same',
                use_bias=False, kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.l2(1e-4))(value)
        x = convolution(x, 16)
        x = L.BatchNormalization()(x)
        x = L.Activation('relu')(x)
        for stage, filters in enumerate((16, 32, 64)):
            for block in range(3):
                stride = 2 if stage > 0 and block == 0 else 1
                shortcut = x
                x = convolution(x, filters, stride)
                x = L.BatchNormalization()(x)
                x = L.Activation('relu')(x)
                x = convolution(x, filters)
                x = L.BatchNormalization()(x)
                if stride == 2:
                    shortcut = convolution(shortcut, filters, stride, size=1)
                    shortcut = L.BatchNormalization()(shortcut)
                x = L.Add()([x, shortcut])
                x = L.Activation('relu', name='features' if stage == 2 and block == 2 else None)(x)
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
    if architecture != 'resnet20':
        x = L.Dropout(0.3)(x)
    logits = L.Dense(10, name='logits')(x)
    model = tf.keras.Model(inputs, logits, name=f'cifar_{architecture}')
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9) if architecture == 'resnet20' else tf.keras.optimizers.Adam(1e-3)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model
