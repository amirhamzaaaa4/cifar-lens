# The 91 percent target

The target was at least 91% accuracy on the official 10,000-image CIFAR-10 test set. The corrected 180-epoch run achieved **93.51%** (95% CI **93.03%–93.98%**) with `smoke_run: false`. See [the completed result](TRAINED_RESULT.md). The smaller residual model remains a separate reference experiment. A single-seed measurement does not guarantee identical performance on further runs.

The corrected ResNet20 candidate uses training-only padded crops and flips, He initialization, L2 regularization, Nesterov momentum SGD, warm-up and cosine learning-rate decay. The current 180-epoch recipe also enables label smoothing, Cutout, mirror averaging and mixed precision. It is trained from scratch.

```bash
python -m lens.train --architecture resnet20 --epochs 180 --batch-size 128 --label-smoothing 0.1 --cutout 8 --tta --mixed-precision --target-accuracy 0.91 --output artifacts/resnet20-corrected-s73
```

Train and tune on the official training set using its existing independent validation split. Select checkpoints using validation loss. Evaluate the official test set after the run. The saved record reports the actual test score and whether the target was met. It must not substitute validation accuracy for test accuracy, alter labels, reuse test images in training, or round a lower score up to 91%.

This long run is intended for a GPU runtime. Free Colab GPU availability varies. CPU execution is supported but can take substantially longer. Do not buy compute or upgrade an account automatically.

The tested notebook configuration uses T4 GPU and runtime version 2026.01 (Python 3.12), matching the project's TensorFlow 2.19 and Keras 3.10 pins. A newer Python runtime may not support these exact pinned versions.

A 180-epoch ResNet20 run and the 50-epoch reference experiments have different optimization budgets; document that difference when comparing them. A miss of the target remains a valid result. Further tuning must use validation evidence. Checkpoints from the earlier in-model crop implementation must not be resumed.

Technical background: https://arxiv.org/abs/1512.03385 and https://arxiv.org/abs/1608.03983. The implementation is new project code, not a claim that published paper results are reproduced.
