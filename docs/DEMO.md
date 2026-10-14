# Three-minute demo

Live demo: https://huggingface.co/spaces/AMIRHAMZAAAA4/cifar-lens. It runs the full corrected ResNet20 weights in the browser with mirror TTA, calibrated scores and Grad-CAM. No GitHub account connection is required.

1. Open CIFAR Lens and identify whether the selected experiment is a smoke run or full experiment.
2. Upload an image from a known class. Explain the 32 × 32 model input, all ten scores, review threshold and Grad-CAM limitations.
3. Open Evaluation. Explain the independent validation split, confusion matrix, weakest class, macro F1 and calibration error.
4. Open Error explorer. Find a confident wrong prediction and explain why confidence alone is insufficient.
5. Explain the project design decisions and how to reproduce the measurements.

For an architecture comparison or multi-seed claim, train baseline, residual and ResNet20 on the same data split and repeat ResNet20 with additional seeds so `lens.compare` reports mean ± std. Record actual scores, hardware and different epoch budgets. The published 93.51% result is explicitly a single full run. Never present a smoke run or the historical score as the new model's benchmark.
