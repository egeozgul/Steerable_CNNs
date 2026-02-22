# Steerable CNNs on Rotated MNIST

A study of rotation-equivariant convolutional neural networks using the `e2cnn` library, trained and evaluated on MNIST and Rotated MNIST datasets.

---

## Overview

This project investigates how **steerable CNNs** — networks built with group-equivariant convolutions — compare to standard CNNs when tested on rotated images. By encoding symmetry priors directly into the network architecture, steerable CNNs generalize to rotations without needing explicit data augmentation.

**Key concepts:**
- **Equivariance** — rotating the input produces a corresponding rotation of the feature maps, rather than arbitrary changes.
- **Invariance** — the final prediction is unchanged regardless of input rotation, achieved here via group pooling.
- **Regular representation** — for a group of order `|G|`, a feature map with `k` regular fields has `k × |G|` channels total.

---

## Results

### C4-Invariant Model vs. Baseline CNN

Both models were trained on a rotated MNIST dataset and evaluated on a held-out rotated test set:

| Model | Rotated MNIST Accuracy |
|---|---|
| Baseline CNN | 26.0% |
| C4-Invariant CNN | **96.3%** |

The C4 model achieves nearly 4× the accuracy of the baseline. The contrast in training behavior is stark — the baseline CNN fails to converge on rotated data, while the C4-invariant model converges smoothly and rapidly:

| Baseline CNN | C4-Invariant CNN |
|---|---|
| ![Baseline training loss](https://raw.githubusercontent.com/egeozgul/Steerable_CNNs/main/Figures/Figure1.png) | ![C4 training loss](https://raw.githubusercontent.com/egeozgul/Steerable_CNNs/main/Figures/Figure2.png) |

### C4 Model Accuracy by Rotation Angle

| Rotation | Accuracy |
|---|---|
| 0° | 84.6% |
| 90° | 69.6% |
| 180° | 63.5% |
| 270° | 74.1% |

A truly C4-invariant network should produce identical accuracy at all four orientations since these are exactly the symmetries it encodes — deviations here reflect the gap between theoretical and practical invariance.

### Symmetry Group Comparison

A unified `EquivariantCNN` class was used to train and evaluate models across six symmetry groups, each trained with matching data augmentation:

| Model | Symmetry | Custom Dataset | Rotated MNIST |
|---|---|---|---|
| D1 | Reflections only | 94.26% | 94.00% |
| C2 | 180° rotations | 96.04% | 47.42% |
| D2 | 180° rotations + reflections | 94.54% | 94.81% |
| D4 | 90° rotations + reflections | 97.36% | **96.78%** |
| C8 | 45° rotations | 99.01% | 42.82% |
| D8 | 45° rotations + reflections | 97.01% | 97.24% |

**Key takeaway:** Dₙ groups (rotation + reflection) generalize well to the rotated MNIST benchmark. Pure cyclic Cₙ models score high on their custom datasets but fail to generalize, indicating that reflections are a crucial inductive bias for this task.

### Training Loss Curves by Symmetry Group

| Model | Training Loss |
|---|---|
| D1 (reflections only) | ![D1](https://raw.githubusercontent.com/egeozgul/Steerable_CNNs/main/Figures/Figure3.png) |
| C2 (180° rotations) | ![C2](https://raw.githubusercontent.com/egeozgul/Steerable_CNNs/main/Figures/Figure4.png) |
| D2 (180° rotations + reflections) | ![D2](https://raw.githubusercontent.com/egeozgul/Steerable_CNNs/main/Figures/Figure5.png) |
| D4 (90° rotations + reflections) | ![D4](https://raw.githubusercontent.com/egeozgul/Steerable_CNNs/main/Figures/Figure6.png) |
| C8 (45° rotations) | ![C8](https://raw.githubusercontent.com/egeozgul/Steerable_CNNs/main/Figures/Figure7.png) |
| D8 (45° rotations + reflections) | ![D8](https://raw.githubusercontent.com/egeozgul/Steerable_CNNs/main/Figures/Figure8.png) |

---

## Model Architectures

### Baseline CNN
A standard CNN with no symmetry constraints — four convolutional blocks (64 channels, 5×5 kernels, stride 2), followed by two linear layers for classification.

### C4-Invariant CNN
Built with `e2cnn`, equivariant under the cyclic group C4 (90° rotations). Uses 64 regular representations of C4 as hidden features, with `GroupPooling` at the end to collapse the group dimension into rotation-invariant scalars.

### General Equivariant CNN
A flexible architecture controlled by two parameters:

```python
EquivariantCNN(N=4, reflections=True)   # D4 — 90° rotations + reflections
EquivariantCNN(N=4, reflections=False)  # C4 — 90° rotations only
EquivariantCNN(N=8, reflections=True)   # D8 — 45° rotations + reflections
```

| Group | N | Reflections |
|---|---|---|
| D1 | 1 | ✓ |
| C2 | 2 | ✗ |
| D2 | 2 | ✓ |
| D4 | 4 | ✓ |
| C8 | 8 | ✗ |
| D8 | 8 | ✓ |

---

## Datasets

| Dataset | Description |
|---|---|
| MNIST | Standard 28×28 handwritten digits, used for training |
| Rotated MNIST | Continuously rotated MNIST (U. Montreal), used for evaluation |
| 90°-rotated MNIST | MNIST with deterministic 0°/90°/180°/270° rotations, used to test discrete invariance |

---

## Usage

```python
# Train a C4-invariant model
model = C4InvariantCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
train(train_loader, model, optimizer, epochs=5)
evaluate(eval_rot_loader, model)

# Train with a different symmetry group
model_d8 = EquivariantCNN(N=8, reflections=True).to(device)
optimizer = optim.Adam(model_d8.parameters(), lr=0.001)
train(train_loader, model_d8, optimizer, epochs=5)
evaluate(eval_rot_loader, model_d8)
```

---

## Dependencies

```bash
pip install torch torchvision e2cnn numpy matplotlib
```

---

## References

- Weiler & Cesa, [General E(2)-Equivariant Steerable CNNs](https://arxiv.org/abs/1911.08251), NeurIPS 2019
- Cohen & Welling, [Group Equivariant Convolutional Networks](https://arxiv.org/abs/1602.07576), ICML 2016
- [e2cnn library](https://github.com/QUVA-Lab/e2cnn)
- [Rotated MNIST dataset](http://www.iro.umontreal.ca/~lisa/icml2007data/)
