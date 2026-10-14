# Three-minute demo

1. Open CIFAR Lens and identify whether the selected experiment is a smoke run or full experiment.
2. Upload an image from a known class. Explain the 32 × 32 model input, all ten scores, review threshold and Grad-CAM limitations.
3. Open Evaluation. Explain the independent validation split, confusion matrix, weakest class, macro F1 and calibration error.
4. Open Error explorer. Find a confident wrong prediction and explain why confidence alone is insufficient.
5. Explain attribution to the original group and AI-assisted portfolio extension.

For a credible GitHub release, train baseline, residual and ResNet20 on the same data split, and run ResNet20 with three seeds so `lens.compare` reports mean ± std. Record actual scores and hardware. Never present the smoke run or the historical score as the new model's benchmark.
