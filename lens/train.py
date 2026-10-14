import argparse
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from lens.data import CLASSES, stratified_split
from lens.metrics import softmax, calibration_error, fit_temperature
from lens.model import build_model


def main():
    parser = argparse.ArgumentParser(description='Train a reproducible CIFAR Lens experiment')
    parser.add_argument('--architecture', choices=['baseline', 'residual'], default='residual')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=73)
    parser.add_argument('--limit', type=int, default=0, help='Training samples for smoke tests; 0 uses all')
    parser.add_argument('--output', type=Path, default=Path('artifacts/residual'))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.limit < 0 or (args.limit and args.limit < 100):
        parser.error('Positive epochs and batch size required; limit must be 0 or >=100')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output folder must be empty; choose a new experiment directory')
    args.output.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    (x, y), (xt, yt) = tf.keras.datasets.cifar10.load_data()
    y, yt = y.reshape(-1), yt.reshape(-1)
    if args.limit:
        rng = np.random.default_rng(args.seed)
        selected = np.concatenate([rng.permutation(np.flatnonzero(y == c))[:args.limit // 10] for c in range(10)])
        x, y = x[selected], y[selected]
    ti, vi = stratified_split(y, seed=args.seed)
    x = x.astype('float32') / 255
    xt = xt.astype('float32') / 255
    model = build_model(args.architecture, args.seed)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6),
                 tf.keras.callbacks.CSVLogger(str(args.output / 'epochs.csv'))]
    history = model.fit(x[ti], y[ti], validation_data=(x[vi], y[vi]), epochs=args.epochs,
                        batch_size=args.batch_size, callbacks=callbacks, verbose=2)
    temperature = fit_temperature(model.predict(x[vi], verbose=0), y[vi])
    logits = model.predict(xt, verbose=0)
    probabilities = softmax(logits, temperature)
    predictions = probabilities.argmax(axis=1)
    ece, reliability = calibration_error(probabilities, yt)
    raw_ece, _ = calibration_error(softmax(logits), yt)
    report = classification_report(yt, predictions, labels=range(10), target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    model.save(args.output / 'model.keras')
    errors = np.flatnonzero(predictions != yt)
    errors = errors[np.argsort(-probabilities[errors].max(axis=1))][:48]
    np.savez_compressed(args.output / 'errors.npz', images=np.uint8(xt[errors] * 255),
                        truth=yt[errors], prediction=predictions[errors], confidence=probabilities[errors].max(axis=1))
    payload = {'schema_version': 1, 'architecture': args.architecture, 'seed': args.seed,
               'smoke_run': bool(args.limit), 'training_samples': len(ti), 'validation_samples': len(vi),
               'test_samples': len(yt), 'temperature': temperature, 'accuracy': report['accuracy'],
               'macro_f1': report['macro avg']['f1-score'], 'ece': ece, 'raw_ece': raw_ece,
               'reliability': reliability, 'classification_report': report,
               'confusion_matrix': confusion_matrix(yt, predictions, labels=range(10)).tolist(),
               'history': {k: [float(v) for v in values] for k, values in history.history.items()},
               'parameters': model.count_params(), 'tensorflow': tf.__version__, 'python': platform.python_version(),
               'model_sha256': hashlib.sha256((args.output / 'model.keras').read_bytes()).hexdigest(),
               'command_config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    (args.output / 'results.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f"Saved experiment to {args.output}; test accuracy {report['accuracy']:.2%}")


if __name__ == '__main__':
    main()
