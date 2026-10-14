import argparse
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from lens.data import CLASSES, stratified_split
from lens.metrics import softmax, calibration_error, fit_temperature, bootstrap_accuracy_ci
from lens.model import ARCHITECTURES, build_model

# Settings that define a run; a resumed run must match them exactly. Keys added
# after the first release default to their old behaviour so older runs resume.
CONFIG_DEFAULTS = {'label_smoothing': 0.0, 'cutout': 0, 'mixed_precision': False}
ERRORS_PER_CLASS = 8


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Train a reproducible CIFAR Lens experiment')
    parser.add_argument('--architecture', choices=ARCHITECTURES, default='residual')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=73)
    parser.add_argument('--limit', type=int, default=0, help='Training samples for smoke tests; 0 uses all')
    parser.add_argument('--output', type=Path, default=Path('artifacts/residual'))
    parser.add_argument('--resume', action='store_true', help='Continue a checkpoint in the output folder')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='e.g. 0.1; 0 disables')
    parser.add_argument('--cutout', type=int, default=0, help='Cutout square size in pixels (resnet20 recipe); 0 disables')
    parser.add_argument('--tta', action='store_true', help='Evaluate with flip test-time augmentation')
    parser.add_argument('--mixed-precision', action='store_true', help='float16 compute on GPU; exported model is float32')
    parser.add_argument('--deterministic', action='store_true', help='Bitwise-repeatable ops; slower on GPU')
    parser.add_argument('--target-accuracy', type=float, default=0.91, help='Recorded goal, never used to alter test results')
    args = parser.parse_args(argv)
    if not 0 < args.target_accuracy <= 1:
        parser.error('Target accuracy must be in (0, 1]')
    if args.epochs < 1 or args.batch_size < 1 or args.limit < 0 or (args.limit and args.limit < 100):
        parser.error('Positive epochs and batch size required; limit must be 0 or >=100')
    if not 0 <= args.label_smoothing < 1:
        parser.error('Label smoothing must be in [0, 1)')
    if not 0 <= args.cutout <= 32:
        parser.error('Cutout must be between 0 and 32 pixels')
    if args.cutout and args.architecture != 'resnet20':
        parser.error('Cutout is part of the resnet20 pipeline recipe')
    return parser, args


def run_config(args):
    return {'architecture': args.architecture, 'seed': args.seed, 'limit': args.limit,
            'batch_size': args.batch_size, 'label_smoothing': args.label_smoothing,
            'cutout': args.cutout, 'mixed_precision': args.mixed_precision}


def main(argv=None):
    parser, args = parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error('Output folder must be empty; choose a new experiment directory')
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / 'training_config.json'
    config = run_config(args)
    if args.resume:
        if not config_path.exists():
            parser.error('Resume requires training_config.json to verify the original data protocol')
        saved_config = json.loads(config_path.read_text())
        # Original runs used deterministic operations; preserve that setting.
        args.deterministic = saved_config.get('deterministic', True)
        saved_config.pop('deterministic', None)
        if {**CONFIG_DEFAULTS, **saved_config} != config:
            parser.error('Resume settings must match the original architecture, seed, sample limit, '
                         'batch size, label smoothing, cutout and precision')
    else:
        config_path.write_text(json.dumps({**config, 'deterministic': args.deterministic}, indent=2), encoding='utf-8')

    tf.keras.utils.set_random_seed(args.seed)
    if args.deterministic:
        tf.config.experimental.enable_op_determinism()
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    (x, y), (xt, yt) = tf.keras.datasets.cifar10.load_data()
    y, yt = y.reshape(-1), yt.reshape(-1)
    if args.limit:
        rng = np.random.default_rng(args.seed)
        selected = np.concatenate([rng.permutation(np.flatnonzero(y == c))[:args.limit // 10] for c in range(10)])
        x, y = x[selected], y[selected]
    ti, vi = stratified_split(y, seed=args.seed)

    pipeline_augment = args.architecture == 'resnet20'
    one_hot = args.label_smoothing > 0
    gpu_available = bool(tf.config.list_physical_devices('GPU'))
    parallel = tf.data.AUTOTUNE if gpu_available and not args.deterministic else 1

    # Data stays uint8 in memory; each element is normalized (and augmented for
    # training) on the fly.
    def batches(images, labels=None, training=False, mirror=False):
        from lens.augment import cutout, pad_crop_flip

        def prepare(image):
            image = tf.cast(image, tf.float32) / 255.
            if training and pipeline_augment:
                image = cutout(pad_crop_flip(image), args.cutout)
            if mirror:
                image = tf.image.flip_left_right(image)
            return image

        def encode(label):
            return tf.one_hot(label, 10) if one_hot else label

        ds = tf.data.Dataset.from_tensor_slices(images if labels is None else (images, labels))
        if training:
            ds = ds.shuffle(min(len(images), 5000), seed=args.seed, reshuffle_each_iteration=True)
        if labels is None:
            ds = ds.map(prepare, num_parallel_calls=parallel)
        else:
            ds = ds.map(lambda image, label: (prepare(image), encode(label)), num_parallel_calls=parallel)
        return ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE if gpu_available else 1)

    initial_epoch = 0
    elapsed_offset = 0
    if args.resume:
        checkpoint = args.output / 'latest.keras'
        if not checkpoint.exists():
            checkpoint = args.output / 'best.keras'
        if not checkpoint.exists() or not (args.output / 'progress.json').exists():
            parser.error('Resume requires a checkpoint and progress record')
        previous = json.loads((args.output / 'progress.json').read_text())
        initial_epoch = previous['epoch']
        elapsed_offset = previous.get('elapsed_seconds', 0)
        model = tf.keras.models.load_model(checkpoint)
        if model.name != f'cifar_{args.architecture}':
            parser.error('Checkpoint architecture does not match')
        if any(isinstance(layer, tf.keras.layers.RandomCrop) for layer in model.layers):
            parser.error('This checkpoint uses the old in-model RandomCrop, which resizes images at '
                         'evaluation time. Start a new experiment with the current code.')
    else:
        model = build_model(args.architecture, args.seed, args.label_smoothing)

    started = time.time()

    class Progress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            (args.output / 'progress.json').write_text(json.dumps({
                'epoch': epoch + 1, 'epoch_budget': args.epochs,
                'elapsed_seconds': elapsed_offset + time.time() - started,
                'metrics': {k: float(v) for k, v in (logs or {}).items()}
            }, indent=2), encoding='utf-8')

    best_checkpoint = tf.keras.callbacks.ModelCheckpoint(str(args.output / 'best.keras'),
                                                         monitor='val_loss', save_best_only=True)
    callbacks = [best_checkpoint,
                 tf.keras.callbacks.ModelCheckpoint(str(args.output / 'latest.keras')),
                 Progress(),
                 tf.keras.callbacks.CSVLogger(str(args.output / 'epochs.csv'), append=args.resume)]
    if args.resume:
        with (args.output / 'epochs.csv').open() as f:
            previous_rows = list(csv.DictReader(f))
        if previous_rows:
            best_checkpoint.best = min(float(row['val_loss']) for row in previous_rows)
    if args.architecture == 'resnet20':
        # Linear warm-up, then cosine decay over the full budget. No early
        # stopping: the low-rate final epochs do most of the refinement.
        def schedule(epoch, current_lr):
            if epoch < 5:
                return 0.1 * (epoch + 1) / 5
            phase = min(1., (epoch - 5) / max(1, args.epochs - 5))
            return float(0.0001 + 0.5 * (0.1 - 0.0001) * (1 + np.cos(np.pi * phase)))
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(schedule))
    else:
        callbacks += [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                      tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6)]

    model.fit(batches(x[ti], y[ti], training=True), validation_data=batches(x[vi], y[vi]),
              epochs=args.epochs, initial_epoch=initial_epoch, callbacks=callbacks, verbose=2)

    # Always evaluate the best validation-loss checkpoint, including an older
    # best checkpoint when a resumed segment never improves on it.
    model = tf.keras.models.load_model(args.output / 'best.keras')

    # Measure the exported float32 weights, so the deployed model matches the
    # recorded predictions and validation-fitted calibration.
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy('float32')
        export = build_model(args.architecture, args.seed, args.label_smoothing)
        export.set_weights(model.get_weights())
        model = export

    def logits_for(images):
        logits = model.predict(batches(images), verbose=0)
        if args.tta:
            logits = (logits + model.predict(batches(images, mirror=True), verbose=0)) / 2
        return np.asarray(logits, dtype=np.float64)

    temperature = fit_temperature(logits_for(x[vi]), y[vi])
    logits = logits_for(xt)
    probabilities = softmax(logits, temperature)
    predictions = probabilities.argmax(axis=1)
    ece, reliability = calibration_error(probabilities, yt)
    raw_ece, _ = calibration_error(softmax(logits), yt)
    report = classification_report(yt, predictions, labels=range(10), target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    ci_low, ci_high = bootstrap_accuracy_ci(predictions == yt, seed=args.seed)

    # The deployed model is always float32 so CPU inference in the app is simple.
    model.save(args.output / 'model.keras')

    # Small authentic gallery for the demo, balanced by true test class.
    from PIL import Image
    gallery = args.output / 'samples'
    gallery.mkdir(exist_ok=True)
    for label, name in enumerate(CLASSES):
        for number, index in enumerate(np.flatnonzero(yt == label)[:3]):
            Image.fromarray(xt[index]).save(gallery / f'{name}_{number}.png')

    # Most confident mistakes per true class, so every class filter has examples.
    confidence = probabilities.max(axis=1)
    errors = []
    for label in range(10):
        wrong = np.flatnonzero((predictions != yt) & (yt == label))
        errors.extend(wrong[np.argsort(-confidence[wrong])][:ERRORS_PER_CLASS])
    errors = np.array(errors, dtype=int)
    errors = errors[np.argsort(-confidence[errors])]
    np.savez_compressed(args.output / 'errors.npz', images=xt[errors], truth=yt[errors],
                        prediction=predictions[errors], confidence=confidence[errors])

    with (args.output / 'epochs.csv').open() as f:
        rows = list(csv.DictReader(f))
    history = {key: [float(row[key]) for row in rows] for key in ['accuracy', 'loss', 'val_accuracy', 'val_loss']}
    payload = {'schema_version': 2, 'architecture': args.architecture, 'seed': args.seed,
               'smoke_run': bool(args.limit), 'training_samples': len(ti), 'validation_samples': len(vi),
               'test_samples': len(yt), 'epochs_trained': len(rows),
               'recipe': {'label_smoothing': args.label_smoothing, 'cutout': args.cutout,
                          'pipeline_augmentation': pipeline_augment, 'tta': args.tta,
                          'mixed_precision': args.mixed_precision, 'deterministic': args.deterministic},
               'tta': args.tta, 'temperature': temperature, 'accuracy': report['accuracy'],
               'accuracy_ci95': [ci_low, ci_high],
               'target_accuracy': args.target_accuracy, 'target_met': report['accuracy'] >= args.target_accuracy,
               'macro_f1': report['macro avg']['f1-score'], 'ece': ece, 'raw_ece': raw_ece,
               'reliability': reliability, 'classification_report': report,
               'confusion_matrix': confusion_matrix(yt, predictions, labels=range(10)).tolist(),
               'history': history,
               'parameters': model.count_params(), 'tensorflow': tf.__version__, 'python': platform.python_version(),
               'training_seconds': elapsed_offset + time.time() - started,
               'resumed': args.resume,
               'hardware': platform.processor(), 'devices': [d.device_type for d in tf.config.list_physical_devices()],
               'model_sha256': hashlib.sha256((args.output / 'model.keras').read_bytes()).hexdigest(),
               'command_config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    (args.output / 'results.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f"Saved experiment to {args.output}; test accuracy {report['accuracy']:.2%} "
          f"(95% CI {ci_low:.2%}–{ci_high:.2%})")


if __name__ == '__main__':
    main()
