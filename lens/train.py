import argparse
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from lens.data import CLASSES, stratified_split
from lens.metrics import softmax, calibration_error, fit_temperature
from lens.model import build_model


def main():
    parser = argparse.ArgumentParser(description='Train a reproducible CIFAR Lens experiment')
    parser.add_argument('--architecture', choices=['baseline', 'residual', 'resnet20'], default='residual')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=73)
    parser.add_argument('--limit', type=int, default=0, help='Training samples for smoke tests; 0 uses all')
    parser.add_argument('--output', type=Path, default=Path('artifacts/residual'))
    parser.add_argument('--resume', action='store_true', help='Continue a checkpoint in the output folder')
    parser.add_argument('--target-accuracy', type=float, default=0.91, help='Recorded goal, never used to alter test results')
    args = parser.parse_args()
    if not 0 < args.target_accuracy <= 1:
        parser.error('Target accuracy must be in (0, 1]')
    if args.epochs < 1 or args.batch_size < 1 or args.limit < 0 or (args.limit and args.limit < 100):
        parser.error('Positive epochs and batch size required; limit must be 0 or >=100')
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error('Output folder must be empty; choose a new experiment directory')
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / 'training_config.json'
    config = {'architecture': args.architecture, 'seed': args.seed,
              'limit': args.limit, 'batch_size': args.batch_size}
    if args.resume:
        if not config_path.exists():
            parser.error('Resume requires training_config.json to verify the original data protocol')
        if json.loads(config_path.read_text()) != config:
            parser.error('Resume settings must match the original architecture, seed, sample limit and batch size')
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    (x, y), (xt, yt) = tf.keras.datasets.cifar10.load_data()
    y, yt = y.reshape(-1), yt.reshape(-1)
    if args.limit:
        rng = np.random.default_rng(args.seed)
        selected = np.concatenate([rng.permutation(np.flatnonzero(y == c))[:args.limit // 10] for c in range(10)])
        x, y = x[selected], y[selected]
    ti, vi = stratified_split(y, seed=args.seed)
    # Keep the dataset in uint8; normalize one batch at a time instead of
    # retaining several full float32 copies during TensorFlow's data conversion.
    def batches(images, labels=None, training=False):
        ds = tf.data.Dataset.from_tensor_slices(images if labels is None else (images, labels))
        if training:
            ds = ds.shuffle(min(len(images), 5000), seed=args.seed)
        ds = ds.batch(args.batch_size)
        if labels is None:
            ds = ds.map(lambda image: tf.cast(image, tf.float32) / 255., num_parallel_calls=1)
        else:
            ds = ds.map(lambda image, label: (tf.cast(image, tf.float32) / 255., label), num_parallel_calls=1)
        return ds.prefetch(1)
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
    else:
        model = build_model(args.architecture, args.seed)
    started = time.time()
    class Progress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            (args.output / 'progress.json').write_text(json.dumps({
                'epoch': epoch + 1, 'epoch_budget': args.epochs,
                'elapsed_seconds': elapsed_offset + time.time() - started,
                'metrics': {k: float(v) for k, v in (logs or {}).items()}
            }, indent=2), encoding='utf-8')
    callbacks = [tf.keras.callbacks.ModelCheckpoint(str(args.output / 'best.keras'), monitor='val_loss', save_best_only=True),
                 tf.keras.callbacks.ModelCheckpoint(str(args.output / 'latest.keras')),
                 Progress(),
                 tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6),
                 tf.keras.callbacks.CSVLogger(str(args.output / 'epochs.csv'), append=args.resume)]
    if args.resume:
        import csv
        with (args.output / 'epochs.csv').open() as f:
            previous_rows = list(csv.DictReader(f))
        if previous_rows:
            callbacks[0].best = min(float(row['val_loss']) for row in previous_rows)
    if args.architecture == 'resnet20':
        # Long SGD schedule: do not stop before the later low-rate refinement.
        def schedule(epoch, current_lr):
            if epoch < 5:
                return 0.1 * (epoch + 1) / 5
            phase = min(1., (epoch - 5) / max(1, args.epochs - 5))
            return float(0.0001 + 0.5 * (0.1 - 0.0001) * (1 + np.cos(np.pi * phase)))
        callbacks = [callback for callback in callbacks if not isinstance(callback,
                     (tf.keras.callbacks.EarlyStopping, tf.keras.callbacks.ReduceLROnPlateau))]
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(schedule))
    model.fit(batches(x[ti], y[ti], training=True), validation_data=batches(x[vi], y[vi]),
              epochs=args.epochs, initial_epoch=initial_epoch, callbacks=callbacks, verbose=2)
    # Always evaluate the best validation-loss checkpoint, including an older
    # best checkpoint when a resumed segment never improves on it.
    model = tf.keras.models.load_model(args.output / 'best.keras')
    temperature = fit_temperature(model.predict(batches(x[vi]), verbose=0), y[vi])
    logits = model.predict(batches(xt), verbose=0)
    probabilities = softmax(logits, temperature)
    predictions = probabilities.argmax(axis=1)
    ece, reliability = calibration_error(probabilities, yt)
    raw_ece, _ = calibration_error(softmax(logits), yt)
    report = classification_report(yt, predictions, labels=range(10), target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    model.save(args.output / 'model.keras')
    # Small authentic gallery for the demo, balanced by true test class.
    from PIL import Image
    gallery = args.output / 'samples'
    gallery.mkdir(exist_ok=True)
    for label, name in enumerate(CLASSES):
        candidates = np.flatnonzero(yt == label)
        for number, index in enumerate(candidates[:3]):
            Image.fromarray(xt[index]).save(gallery / f'{name}_{number}.png')
    errors = np.flatnonzero(predictions != yt)
    errors = errors[np.argsort(-probabilities[errors].max(axis=1))][:48]
    np.savez_compressed(args.output / 'errors.npz', images=xt[errors],
                        truth=yt[errors], prediction=predictions[errors], confidence=probabilities[errors].max(axis=1))
    import csv
    with (args.output / 'epochs.csv').open() as f:
        rows = list(csv.DictReader(f))
    history = {key: [float(row[key]) for row in rows] for key in ['accuracy', 'loss', 'val_accuracy', 'val_loss']}
    payload = {'schema_version': 1, 'architecture': args.architecture, 'seed': args.seed,
               'smoke_run': bool(args.limit), 'training_samples': len(ti), 'validation_samples': len(vi),
               'test_samples': len(yt), 'temperature': temperature, 'accuracy': report['accuracy'],
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
    print(f"Saved experiment to {args.output}; test accuracy {report['accuracy']:.2%}")


if __name__ == '__main__':
    main()
