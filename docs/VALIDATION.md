# Validation record

Verified on 14 October 2026 with Python 3.12, TensorFlow 2.19.0 and Keras 3.10.0 on Windows CPU.

- Six data, numerical and model integration tests passed. Both architectures completed a training batch and produced finite Grad-CAM maps.
- The dashboard test passed, including filtering real saved errors by true class.
- A complete one-epoch residual smoke experiment ran with 900 training, 100 validation and all 10,000 test images. Model, metrics, calibration, curves and error images were saved.
- Smoke accuracy was 14.36%. This verifies execution only and is not a useful classifier or a benchmark improvement.
- The CIFAR-10 archive was obtained from a Zenodo mirror and matched the canonical SHA-256 `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce` before use. Ordinary project training uses the TensorFlow Keras dataset loader.

Full baseline/residual training, cross-seed comparisons, deployed hosting and browser image-upload validation remain outstanding. No improvement over the historical 85.07% result is claimed.

## Changes after the first validation (portfolio upgrade)

- **Bug fixed:** ResNet20 combined padding and training-dependent `RandomCrop` inside the model, so inference did not follow the training preprocessing. In the installed Keras 3.10 source, the inference branch returns the padded image unchanged; other versions may resize it. Augmentation now runs in the `tf.data` pipeline for training only. Resuming an old ResNet20 checkpoint is refused.
- Uploads are centre-cropped to a square before resizing instead of being stretched.
- New recipe options: label smoothing, Cutout, flip TTA, mixed precision (float32 export) and opt-in determinism.
- Results now include a bootstrap 95% accuracy interval and the per-class confident errors. `lens.compare` groups seeds into mean ± std and accepts different epoch budgets.
- The app caches the checksum-verified model and the Grad-CAM graph, and can download a published experiment (`LENS_EXPERIMENT_URL`).
- Docker image runs as non-root with a health check. CI adds a pyflakes-level `ruff` check.

The supplied update was initially checked without TensorFlow. After integration on Windows with Python 3.12, TensorFlow 2.19 and Keras 3.10:

- 20 data, metric, comparison, packaging, model and saved-inference tests passed. All three architectures trained a batch and produced finite Grad-CAM maps; label smoothing and training-only augmentation were exercised.
- The dashboard test passed, including its class filter. The updated packaging checks also passed for nested experiment archives and Windows path rejection.
- Ruff passed for the maintained source. The historical group notebook is excluded to preserve its archived code.
- A corrected one-epoch ResNet20 smoke run completed with 900 training, 100 validation and 10,000 test images, label smoothing 0.1, Cutout 8 and flip TTA. It saved checkpoints, exported weights, calibration and error examples. Its 10.00% test accuracy is an execution check, not a benchmark.
- Mixed-precision GPU training and the full accuracy target still require a completed GPU run. Robustness, unknown-image detection and multiple-seed results remain unmeasured.

The expanded suite passed 22 model and utility tests, including mixed-precision training/float32 export and saved corrected ResNet20 inference. The dashboard hit its 60-second cold-start limit in the combined run while CPU training was active; with a writable plotting cache and a 180-second test limit, it passed in 48.78 seconds. This checks dashboard execution, not browser interaction or hosted performance.

## Completed model and public browser demo — 14 October 2026

The full corrected ResNet20 GPU run completed all 180 epochs. Its record confirms `smoke_run: false`, `architecture: resnet20`, 45,000 training images, 5,000 validation images and 10,000 test images. Test accuracy is **93.51%**, 95% bootstrap interval **93.03%–93.98%**, macro F1 **0.93498555**, and calibrated ECE **0.01410733** (raw ECE **0.06410152**). The best validation-loss checkpoint was exported to float32 before measurement. Model and release archive checksums were verified after transfer to the laptop.

The trained experiment is published in [GitHub Release v1.0.0](https://github.com/amirhamzaaaa4/cifar-lens/releases/tag/v1.0.0). The historical notebook's test-as-validation score remains non-comparable; this is a single-seed result.

The [Hugging Face static demo](https://huggingface.co/spaces/AMIRHAMZAAAA4/cifar-lens) uses the actual float32 weights and runs locally in the browser. Its weights checksum is checked before prediction. Thirty identical CIFAR-10 test images were compared with Python: **30/30 class matches**, maximum calibrated-score difference **2.38e-7**, and maximum logit difference **2.80e-6** on the tested WebGL browser. The check runs again on each page initialization and prevents prediction if it fails.

Browser interaction verified a real uploaded `ship_1.png` (ship, 96.33% calibrated confidence), sample selection, displayed Grad-CAM, and the cat filter containing eight authentic errors. Screenshots and an animated sequence of actual app views are stored under `evidence/trained/`. The Grad-CAM formula is exact for this model's global-average-pooling and linear classifier head; it explains the original view while scores use mirror TTA.

Uploaded photos are centre-cropped and normalized consistently. Browser canvas resize is not claimed to be bit-identical to Pillow bicubic. Browser parity is measured on identical stored CIFAR-10 pixels, not independently decoded/resized photos. Additional seeds, robustness and unknown-object detection remain unmeasured. Baseline and residual reference training are separate optional experiments.
