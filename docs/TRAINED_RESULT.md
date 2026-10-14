# Completed ResNet20 experiment

Test accuracy: **93.51%** (95% bootstrap interval: 93.03%–93.98%).
Macro F1: 0.9350. Calibrated ECE: 0.0141; uncalibrated ECE: 0.0641.

The run completed 180 epochs. Model parameters: 274,042.
The 91% target was met by this single-seed experiment.

Training used 45,000 images and a separate 5,000-image validation split. The best validation-loss checkpoint was exported to float32 before evaluation. Mirror TTA and validation-fitted temperature scaling match the deployed app's inference settings. The official 10,000-image test set was used after checkpoint selection.

Model SHA-256: `12aa468bc2fc9fcb8c10091653019ad6d3a27f429e49fe8f2069f97f70b39af2`.

This result is not a multi-seed average. The historical assignment score used the test set for validation and is not directly comparable.
