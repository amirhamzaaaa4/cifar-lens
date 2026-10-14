---
title: CIFAR Lens
emoji: 🔎
colorFrom: green
colorTo: blue
license: mit
sdk: static
app_file: index.html
pinned: false
---

# CIFAR Lens browser demo

Real ResNet20 weights, mirror-image test-time averaging, validation-fitted temperature scaling and Grad-CAM. Photos are processed locally in the visitor's browser.

The 93.51% result is measured by the Python experiment on 10,000 held-out CIFAR-10 images. Browser inference verifies the weights checksum and compares 30 authentic test images with their Python predictions at startup.

Uploaded photos are centre-cropped to a square, resized to 32×32 and divided by 255. Browser canvas resizing can differ slightly from Pillow bicubic; the parity check uses identical stored CIFAR-10 pixels.

Grad-CAM uses the exact pooled-gradient formula for this model's global-average-pooling and linear classifier head, then normalizes the positive class activation map. Scores include mirror TTA; the heatmap explains the original view.

Source and trained-model release: https://github.com/amirhamzaaaa4/cifar-lens

Developed by Amir Hamza. Licensed under the [MIT License](LICENSE).
