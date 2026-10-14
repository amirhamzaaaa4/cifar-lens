# The 91 percent target

The target is at least 91% accuracy on the official 10,000-image CIFAR-10 test set. It is a goal, not a measured result or a guarantee. The current smaller residual model remains a reference experiment.

The new ResNet20 candidate adds a deeper CIFAR-scale residual architecture, padded random crops, horizontal flips, He initialization, L2 regularization, momentum SGD, a five-epoch warm-up and cosine learning-rate decay. It is trained from scratch. A 200-epoch budget allows refinement after the learning rate decreases.

```bash
python -m lens.train --architecture resnet20 --epochs 200 --batch-size 128 --target-accuracy 0.91 --output artifacts/resnet20-target91
```

Train and tune on the official training set using its existing independent validation split. Select checkpoints using validation loss. Evaluate the official test set after the run. The saved record reports the actual test score and whether the target was met. It must not substitute validation accuracy for test accuracy, alter labels, reuse test images in training, or round a lower score up to 91%.

This long run is intended for a GPU runtime. Free Colab GPU availability varies. CPU execution is supported but can take substantially longer. Do not buy compute or upgrade an account automatically.

The tested notebook configuration uses T4 GPU and runtime version 2026.01 (Python 3.12), matching the project's TensorFlow 2.19 and Keras 3.10 pins. A newer Python runtime may not support these exact pinned versions.

A 200-epoch ResNet20 run and the 50-epoch reference experiments have different optimization budgets; document that difference when comparing them. A miss of the target remains a valid result. Further architecture or optimizer tuning must use validation evidence and preserve an untouched final evaluation protocol.

Technical background: https://arxiv.org/abs/1512.03385 and https://arxiv.org/abs/1608.03983. The implementation is new project code, not a claim that published paper results are reproduced.
