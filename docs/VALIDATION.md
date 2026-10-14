# Validation record

Verified on 14 October 2026 with Python 3.12, TensorFlow 2.19.0 and Keras 3.10.0 on Windows CPU.

- Six data, numerical and model integration tests passed. Both architectures completed a training batch and produced finite Grad-CAM maps.
- The dashboard test passed, including filtering real saved errors by true class.
- A complete one-epoch residual smoke experiment ran with 900 training, 100 validation and all 10,000 test images. Model, metrics, calibration, curves and error images were saved.
- Smoke accuracy was 14.36%. This verifies execution only and is not a useful classifier or a benchmark improvement.
- The CIFAR-10 archive was obtained from a Zenodo mirror and matched the canonical SHA-256 `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce` before use. Ordinary project training uses the TensorFlow Keras dataset loader.

Full baseline/residual training, cross-seed comparisons, deployed hosting and browser image-upload validation remain outstanding. No improvement over the historical 85.07% result is claimed.
