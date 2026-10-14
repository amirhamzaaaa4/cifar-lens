"""Export float32 weights and authentic Python reference predictions for the static demo."""
import hashlib
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

import argparse
parser = argparse.ArgumentParser(description='Export a verified ResNet20 experiment for browser inference')
parser.add_argument('--experiment', type=Path, required=True)
parser.add_argument('--output', type=Path, default=Path('web'))
args = parser.parse_args()
source = args.experiment
target = args.output
target.mkdir(exist_ok=True)
r = json.loads((source / 'results.json').read_text())
assert r['architecture'] == 'resnet20' and not r['smoke_run'], 'A full corrected ResNet20 experiment is required'
assert hashlib.sha256((source / 'model.keras').read_bytes()).hexdigest() == r['model_sha256'], 'Model checksum mismatch'
model = tf.keras.models.load_model(source / 'model.keras', compile=False)
graph, chunks, offset = [], [], 0
for layer in model.layers:
    inputs = [] if isinstance(layer, tf.keras.layers.InputLayer) else [t._keras_history.operation.name for t in (layer.input if isinstance(layer.input, list) else [layer.input])]
    weights = []
    for array in layer.get_weights():
        array = np.asarray(array, dtype='<f4')
        weights.append({'shape': list(array.shape), 'offset': offset, 'length': array.size})
        chunks.append(array.tobytes())
        offset += array.size
    graph.append({'name': layer.name, 'type': type(layer).__name__, 'config': layer.get_config(), 'inputs': inputs, 'weights': weights})
binary = b''.join(chunks)
(target / 'weights.bin').write_bytes(binary)
manifest = {'layers': graph, 'output': model.output._keras_history.operation.name, 'features': 'features', 'weights_sha256': hashlib.sha256(binary).hexdigest(), 'source_model_sha256': r['model_sha256']}
(target / 'model.json').write_text(json.dumps(manifest), encoding='utf-8')
from PIL import Image
samples = []
for p in sorted((source / 'samples').glob('*.png')):
    image = np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8)
    batch = image[None].astype(np.float32) / 255
    original = model(batch, training=False).numpy()[0]
    mirror = model(np.ascontiguousarray(batch[:, :, ::-1]), training=False).numpy()[0]
    tta = (original+mirror)/2
    probs = np.exp((tta-tta.max())/r['temperature'])
    probs /= probs.sum()
    samples.append({'name': p.stem.rsplit('_', 1)[0], 'id': p.stem, 'pixels': image.reshape(-1).tolist(), 'logits': original.tolist(), 'tta_logits': tta.tolist(), 'probabilities': probs.tolist()})
(target / 'samples.json').write_text(json.dumps(samples), encoding='utf-8')
errors = np.load(source / 'errors.npz')
(target / 'errors.json').write_text(json.dumps([{k: (errors[k][i].tolist() if k == 'images' else float(errors[k][i])) for k in errors.files} for i in range(len(errors['truth']))]), encoding='utf-8')
(target / 'results.json').write_text(json.dumps(r), encoding='utf-8')
print('Exported browser model:', len(binary), 'bytes;', len(samples), 'parity examples')
