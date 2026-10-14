# ◉ CIFAR Lens

![CIFAR Lens computer vision lab](evidence/banner.svg)

[![Tests](https://github.com/amirhamzaaaa4/cifar-lens/actions/workflows/tests.yml/badge.svg)](https://github.com/amirhamzaaaa4/cifar-lens/actions/workflows/tests.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amirhamzaaaa4/cifar-lens/blob/main/notebooks/Train_on_Colab.ipynb)

### See the prediction. Explore the evidence.

An interactive computer-vision lab for CIFAR-10. Upload a photo and see all ten class scores, a Grad-CAM heatmap of the regions behind the prediction, calibrated confidence, and the model's most confident mistakes. It also includes a reproducible training pipeline whose test set is never used for model selection.

**TensorFlow / Keras · Streamlit · Grad-CAM · calibration · reproducible experiments**

The live demo awaits a completed trained-model release. Current target: **91% held-out test accuracy**.

<!-- Add a short GIF here: upload → prediction → Grad-CAM → error explorer -->

## Results

All new models are trained on 45,000 images, selected on a separate 5,000-image validation split, and evaluated once on the official 10,000-image test set.

| Model | Test accuracy (95% CI) | Macro F1 | Calibration error (ECE) | Parameters |
|---|---:|---:|---:|---:|
| Original group notebook (2025) | 85.07%\* | 0.849 | — | — |
| Baseline CNN | _run pending_ | | | |
| Residual CNN | _run pending_ | | | |
| ResNet20 + Cutout + label smoothing + flip TTA | _run pending_ | | | ≈0.27 M |

\* The original notebook also used the test set for validation, so its score is optimistic and not directly comparable.

Fill this table from `python -m lens.compare …` (see below). It writes `evidence/benchmark/comparison.csv` and an accuracy chart.

## What makes it different

- **Clean evaluation:** seeded stratified train/validation split. Early stopping, checkpoint choice and temperature calibration only ever see validation data.
- **Uncertainty you can check:** temperature scaling, a reliability plot, expected calibration error, and a bootstrap 95% confidence interval on test accuracy.
- **Explanations with caveats:** Grad-CAM from the live model, plus an error explorer showing the most confident mistakes for every class.
- **Reproducible runs:** every experiment records its seed, split sizes, recipe, library versions, hardware and a SHA-256 of the saved model. Interrupted runs resume safely, and multi-seed runs are compared as mean ± std.
- **Tested:** data-integrity, numerical, model, packaging and dashboard tests run in GitHub Actions.

## Run the app

Use Python 3.12.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m streamlit run app.py
```

The dashboard opens without a trained model and shows the historical results. To get live predictions, train an experiment or download a published one:

```bash
python -m lens.fetch <release-zip-url> --output artifacts/resnet20
```

## Train

For GPU training, the easiest route is the [Colab notebook](notebooks/Train_on_Colab.ipynb). It saves checkpoints to Google Drive so a disconnected session can resume.

```bash
# Short pipeline check (not a benchmark)
python -m lens.train --epochs 1 --limit 1000 --output artifacts/smoke

# Candidate recipe: ResNet20 on a GPU; runtime depends on available hardware
python -m lens.train --architecture resnet20 --epochs 180 --label-smoothing 0.1 --cutout 8 \
    --tta --mixed-precision --seed 73 --output artifacts/resnet20-s73

# Reference models
python -m lens.train --architecture baseline --epochs 50 --output artifacts/baseline
python -m lens.train --architecture residual --epochs 50 --output artifacts/residual

# Resume an interrupted run (same settings + --resume)
python -m lens.train --architecture resnet20 --epochs 180 --label-smoothing 0.1 --cutout 8 \
    --tta --mixed-precision --seed 73 --output artifacts/resnet20-s73 --resume

# Compare (repeat the ResNet20 run with --seed 1 and --seed 2 for mean ± std)
python -m lens.compare artifacts/baseline artifacts/residual artifacts/resnet20-s*
python -m pytest -q
```

Each run writes `model.keras`, `results.json`, `epochs.csv`, `errors.npz` and sample images to its own folder. Folders must be empty to prevent overwrites. Weights stay out of Git: publish an experiment folder as a zip in a GitHub Release.

## Deploy

Hugging Face Spaces (Docker) or any container host: build the included `Dockerfile` and set `LENS_EXPERIMENT_URL` to your release zip. The app downloads it on startup and verifies the model checksum. See [hosting notes](docs/HOSTING.md).

## Project structure

```text
app.py              Streamlit lab
lens/model.py       Baseline, residual CNN and ResNet20
lens/augment.py     Training-only pad-crop-flip and Cutout
lens/train.py       Reproducible training, calibration and evaluation
lens/metrics.py     Softmax, ECE, temperature scaling, bootstrap CI, flip TTA
lens/explain.py     Grad-CAM
lens/compare.py     Multi-run, multi-seed comparison
lens/fetch.py       Download and verify published experiments
notebooks/          Colab training notebook, walkthrough, original group notebook
docs/               Model card, validation record, provenance, demo guide
tests/              Unit, integration and dashboard tests
```

## Design decisions

- **Why a separate validation split?** The original notebook selected its model on the test set, which inflates the reported score. Here the test set is used once.
- **Why augmentation in the data pipeline?** Keras `RandomCrop` resizes the whole padded image at inference time. Putting it inside the model would shrink every evaluated image, so augmentation happens in `tf.data`, for training only.
- **Why temperature scaling?** Networks trained with cross-entropy tend to be overconfident. One validation-fitted temperature fixes most of that without changing any prediction.
- **Why logits plus softmax instead of the assignment's ReLU output?** Valid probabilities are needed for calibration and the confidence threshold.

## Limitations

32 × 32 inputs and ten fixed classes. Every upload is forced into one of them, and there is no unknown-object detector yet. Grad-CAM shows coarse correlation, not causation. Not for consequential decisions. See the [model card](docs/MODEL_CARD.md).

## Credits

Original CIFAR-10 assignment (COIT12213, 2025) by **Rory O'Brien, Amir Hamza and MD Suyaib Rahman**; the historical notebook is kept in `notebooks/` with student IDs removed. The portfolio extension (app, pipeline, tests, docs) was developed by Amir Hamza with AI coding assistance. See [provenance](docs/PROVENANCE.md).

No license has been applied yet; it requires agreement from all original authors.

## References

- He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), 2016
- DeVries & Taylor, [Improved Regularization of CNNs with Cutout](https://arxiv.org/abs/1708.04552), 2017
- Guo et al., [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599), 2017
- Selvaraju et al., [Grad-CAM](https://arxiv.org/abs/1610.02391), 2017
- [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
