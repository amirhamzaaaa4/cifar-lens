# CIFAR Lens model card

## Intended use

Educational exploration of ten-class image classification, uncertainty and error analysis. Not suitable for consequential decisions.

## Data and protocol

Official CIFAR-10: 50,000 training and 10,000 test images, RGB, 32 × 32. A seeded stratified split reserves 10% of official training images for validation. Test labels do not control callbacks or temperature selection. Ordinary runs use integer labels and sparse cross-entropy; label-smoothed runs use one-hot labels and categorical cross-entropy. Class probabilities use softmax over logits.

## Models

- **Baseline:** three convolution layers with pooling. Adam optimizer, early stopping on validation loss.
- **Residual CNN:** three stages with 32/64/128 channels, shortcut connections, batch normalization, L2 regularization, dropout, and in-model horizontal flips and translations (both are identity at inference).
- **ResNet20** (He et al., 2016): 3 × 3 residual blocks with 16/32/64 channels and about 0.27 M parameters. Nesterov SGD with 5-epoch warm-up and cosine decay, L2 1e-4. Training-only pad-4 random crop, horizontal flip and optional Cutout in the `tf.data` pipeline, optional label smoothing, and optional flip test-time augmentation. Augmentation is kept out of the saved model, so inference always sees the unmodified image.

All use global average pooling and float32 logits. No architecture is claimed to beat another until measured under the same split.

## Evaluation

Accuracy with a 2,000-sample bootstrap 95% interval, macro F1, class precision/recall/F1, confusion matrix and 15-bin expected calibration error. Temperature is selected using validation negative log likelihood over a fixed grid. Calibrated probability scores are not guarantees. The error explorer saves the 8 most confident test mistakes per true class, which is a selected sample and not representative of all errors. When TTA is enabled, temperature is fitted on TTA validation logits, and the app applies the same TTA.

## Known limitations

Fixed categories force every image into a known class. High-confidence out-of-distribution failures remain possible. Grad-CAM is a coarse attribution tool, not causal evidence or a correctness guarantee. Resizing arbitrary uploads can remove critical detail. Bitwise determinism is opt-in (`--deterministic`) because it slows GPU training; cross-platform bitwise reproducibility is not guaranteed.

## Evidence status

Original notebook records 85.07% accuracy and macro F1 0.8488 but reused test data as validation. New artifacts contain explicit sample counts, seed, architecture, versions and model checksum. Smoke runs are labeled and excluded from benchmark claims.
