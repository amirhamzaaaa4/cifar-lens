# ◉ CIFAR Lens

![CIFAR Lens computer vision lab](evidence/banner.svg)

### See the prediction. Explore the evidence.

An interactive computer-vision lab built from a university CIFAR-10 CNN project. Upload an image, inspect ten class scores, explore Grad-CAM, and investigate confident mistakes.

**TensorFlow · Streamlit · reproducible experiments · calibration · model interpretability**

**Current goal: at least 91% held-out test accuracy.** A new ResNet20 candidate and a [GPU training notebook](notebooks/CIFAR_Lens_Target91_Colab.ipynb) are included. This is a training target, not a measured result. See the [accuracy protocol](docs/ACCURACY_TARGET.md).

## Why this project exists

A model score alone tells little about how a classifier behaves. CIFAR Lens connects predictions to evaluation evidence: per-class performance, model confidence, visual attribution, and error examples. It also fixes the original experiment's test/validation overlap.

## Run locally

Use Python 3.12. Windows CPU is supported; for GPU training use a compatible Linux/WSL or Colab TensorFlow environment.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The dashboard opens with historical evaluation evidence. Add a completed experiment to enable predictions, Grad-CAM and error exploration.

```bash
# Quick pipeline check; not an accuracy benchmark
python -m lens.train --epochs 1 --limit 1000 --output artifacts/smoke

# Full experiments under the same protocol
python -m lens.train --architecture baseline --epochs 50 --output artifacts/baseline
python -m lens.train --architecture residual --epochs 50 --output artifacts/residual
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Training now saves both the best and latest checkpoint, and normalizes one batch at a time to reduce memory use. Continue an interrupted experiment with the same architecture, seed, limit and batch size:

```bash
python -m lens.train --architecture residual --epochs 50 --batch-size 128 --output artifacts/residual --resume
python -m lens.compare artifacts/baseline artifacts/residual
```

The app includes real dataset examples when a completed experiment provides them. See [hosting instructions](docs/HOSTING.md) for Streamlit Community Cloud and container deployment.

Training downloads CIFAR-10 on first use. On modest hardware, start with batch size 64. Each experiment directory must be empty to prevent accidental overwrites. Training writes model.keras, results.json, epochs.csv and errors.npz. Keep weights out of ordinary Git commits; distribute trained weights through release assets.

## Features

- Image upload, ten-class scores and downloadable prediction records
- Grad-CAM visual attribution computed from the active model
- Validation-fitted temperature scaling and confidence review control
- Training curves, confusion matrix, per-class metrics and reliability plot
- Explorer for the model's most confident test errors
- Seeded stratified split, separate final test evaluation and model checksum
- Smaller baseline and residual CNN with a consistent evaluation protocol
- ResNet20 and a T4 Colab notebook for longer GPU experiments
- Tests and GitHub Actions workflow

## Results

| Experiment | Accuracy | Macro F1 | Status |
|---|---:|---:|---|
| Original supplied notebook | 85.07% | 0.8488 | Historical; test set also used for validation |
| New baseline | Pending full training | Pending | No performance claim |
| New residual CNN | Pending full training | Pending | No performance claim |
| ResNet20 | Training target: 91% | Pending | 200-epoch GPU recipe |

A higher score is not promised. Full experiments need to establish whether the residual architecture improves on the new baseline. Comparison with the old score is limited by the changed evaluation protocol.

## Structure

```text
app.py              Interactive lab
lens/               Data, models, training, metrics and Grad-CAM
notebooks/          Clean walkthrough and historical group notebook
artifacts/          Local experiment outputs (ignored by Git)
evidence/           Extracted historical results and original figures
docs/               Model card, provenance and demo guide
tests/              Data integrity, numeric and model integration tests
```

## Design decisions

- Reserve validation images from the training set so model selection and calibration leave the test set untouched.
- Return logits, then fit temperature on validation data to make confidence measurable alongside accuracy.
- Cache verified weights and the Grad-CAM probe to keep repeated predictions responsive.
- Record training budgets alongside scores: a longer experiment is not evidence that architecture alone caused an improvement.

## Credits

Original CIFAR-10 assignment by **Rory O'Brien, Amir Hamza and MD Suyaib Rahman**, COIT12213, 2025, using the course starter notebook. This portfolio extension was developed with OpenAI Codex assistance. The archived group notebook retains its technical cells and results, with student IDs removed; its original AI declaration applies to the original submission. See [provenance](docs/PROVENANCE.md).

No license is assigned to the original group work without the contributors' agreement. See [provenance](docs/PROVENANCE.md) before publishing.

## Technical references

- [TensorFlow training and evaluation](https://www.tensorflow.org/guide/keras/training_with_built_in_methods)
- [Keras Grad-CAM example](https://keras.io/examples/vision/grad_cam/)
- [Streamlit resource caching](https://docs.streamlit.io/develop/concepts/architecture/caching)
- [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

See [model card](docs/MODEL_CARD.md) and [demo guide](docs/DEMO.md).
