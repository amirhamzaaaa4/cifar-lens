# CIFAR Lens model card

## Intended use

Educational exploration of ten-class image classification, uncertainty and error analysis. Not suitable for consequential decisions.

## Data and protocol

Official CIFAR-10: 50,000 training and 10,000 test images, RGB, 32 × 32. A seeded stratified split reserves 10% of official training images for validation. Test labels do not control callbacks or temperature selection. Sparse integer labels and logits are paired with sparse cross-entropy. The assignment specification's ReLU output instruction is not used because this portfolio version needs valid class probabilities, obtained by softmax over logits.

## Models

Baseline: three convolution layers and pooling. Residual: three stages with 32/64/128 channels, shortcut connections, batch normalization, L2 regularization, dropout, horizontal flips and translations. Both use global average pooling. Neither is claimed to exceed the original score until fully trained and evaluated.

ResNet20: three stages with 16/32/64 channels and three two-convolution residual blocks per stage. Padded random crops and horizontal flips augment training images. SGD with momentum uses five warmup epochs followed by cosine learning-rate decay. The GPU notebook requests 200 epochs. The 91% test-accuracy target remains unmeasured until evaluation completes.

## Evaluation

Accuracy, macro F1, class precision/recall/F1, confusion matrix and 15-bin expected calibration error. Temperature is selected using validation negative log likelihood over a fixed grid. Calibrated probability scores are not guarantees. New runs save up to 12 confident mistakes per true class; older artifacts contain the 48 most confident errors overall. Both are selected samples, not representative of all errors. Comparisons report epoch budgets and completed epochs; different budgets do not isolate architecture effects.

## Known limitations

Fixed categories force every image into a known class. High-confidence out-of-distribution failures remain possible. Grad-CAM is a coarse attribution tool, not causal evidence or a correctness guarantee. Uploaded images are center-cropped before resizing, which can remove objects near their edges. Deterministic operations are optional with --deterministic; older checkpoints preserve their original deterministic setting when resumed. Cross-platform bitwise reproducibility is not guaranteed.

## Evidence status

Original notebook records 85.07% accuracy and macro F1 0.8488 but reused test data as validation. New artifacts contain explicit sample counts, seed, architecture, versions and model checksum. Smoke runs are labeled and excluded from benchmark claims.
