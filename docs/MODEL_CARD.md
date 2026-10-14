# CIFAR Lens model card

## Intended use

Educational exploration of ten-class image classification, uncertainty and error analysis. Not suitable for consequential decisions.

## Data and protocol

Official CIFAR-10: 50,000 training and 10,000 test images, RGB, 32 × 32. A seeded stratified split reserves 10% of official training images for validation. Test labels do not control callbacks or temperature selection. Sparse integer labels and logits are paired with sparse cross-entropy. The assignment specification's ReLU output instruction is not used because this portfolio version needs valid class probabilities, obtained by softmax over logits.

## Models

Baseline: three convolution layers and pooling. Residual: three stages with 32/64/128 channels, shortcut connections, batch normalization, L2 regularization, dropout, horizontal flips and translations. Both use global average pooling. Neither is claimed to exceed the original score until fully trained and evaluated.

## Evaluation

Accuracy, macro F1, class precision/recall/F1, confusion matrix and 15-bin expected calibration error. Temperature is selected using validation negative log likelihood over a fixed grid. Calibrated probability scores are not guarantees. The error explorer saves the 48 most confident test mistakes and therefore is a selected sample, not representative of all errors.

## Known limitations

Fixed categories force every image into a known class. High-confidence out-of-distribution failures remain possible. Grad-CAM is a coarse attribution tool, not causal evidence or a correctness guarantee. Resizing arbitrary uploads can remove critical detail. GPU determinism is requested but cross-platform bitwise reproducibility is not guaranteed.

## Evidence status

Original notebook records 85.07% accuracy and macro F1 0.8488 but reused test data as validation. New artifacts contain explicit sample counts, seed, architecture, versions and model checksum. Smoke runs are labeled and excluded from benchmark claims.
