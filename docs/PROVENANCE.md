# Project history

CIFAR Lens is maintained by Amir Hamza.

The earlier CIFAR-10 notebook is retained as a historical experiment. Its 85.07% score used the test set for validation and is not directly comparable with the new independent evaluation.

The current project adds an independent validation split, configurable architectures, training-only augmentation, model saving, verified experiment metadata, calibrated confidence, Grad-CAM, an interactive app, browser inference, tests and documentation.

The corrected ResNet20 run achieved 93.51% test accuracy on 10,000 held-out images after 180 training epochs. See the model card and full evaluation record for the recipe, measured limitations and reproducibility details.

The course specification provides historical context. The current implementation uses logits and softmax with validation-fitted temperature scaling. Course documents, student IDs and private assessment screenshots are excluded from the public repository.
