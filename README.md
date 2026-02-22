# Steerable CNNs on Rotated MNIST

A study of rotation-equivariant convolutional neural networks using the `e2cnn` library, trained and evaluated on MNIST and Rotated MNIST datasets.

---

## Overview

This notebook investigates how **steerable CNNs** — networks built with group-equivariant convolutions — compare to standard CNNs when tested on rotated images. By encoding symmetry priors directly into the network architecture, steerable CNNs generalize better to transformations they were never explicitly trained on.

---

## Architectures

### Baseline CNN (`CNNBaseline`)

A standard convolutional network with no symmetry constraints.

| Layer | Details |
|-------|---------|
| Conv1 | 1 → 64 channels, 5×5 kernel, stride 2, padding 2 |
| Conv2 | 64 → 64 channels, 5×5 kernel, stride 2, padding 2 |
| Conv3 | 64 → 64 channels, 5×5 kernel, stride 2, padding 2 |
| Conv4 | 64 → 64 channels, 5×5 kernel, stride 2, padding 2 |
| Linear1 | 256 → 128, ReLU |
| Linear2 | 128 → 10, log-softmax |

Activations: ReLU throughout. Total spatial downsampling: 28×28 → 2×2 → flattened to 256.

---

### C4-Invariant CNN (`C4InvariantCNN`)

Built using the `e2cnn` library. Equivariant under the **cyclic group C4** (discrete 90° rotations).

**Key design choices:**

- **Input type:** Trivial representation (scalar field) — standard grayscale image
- **Hidden type:** 64 regular representations of C4 (each field has 4 channels, one per group element)
- **Pooling to invariance:** `GroupPooling` collapses the group dimension, producing rotation-invariant features

| Layer | Details |
|-------|---------|
| R2Conv 1 | Trivial → 64 × Regular(C4), 5×5, stride 2 |
| R2Conv 2 | 64 × Regular(C4) → same, 5×5, stride 2 |
| R2Conv 3 | same → same, 5×5, stride 2 |
| R2Conv 4 | same → same, 5×5, stride 2 |
| PointwiseAvgPool | Spatial pooling to 1×1 |
| GroupPooling | C4 group dimension → invariant scalar (64 × 4 = 256 features) |
| Linear1 | 256 → 128, ReLU |
| Linear2 | 128 → 10, log-softmax |

---

### General Equivariant CNN (`EquivariantCNN`)

A flexible architecture supporting multiple symmetry groups:

| Group | N | Reflections | Symmetry |
|-------|---|-------------|---------|
| D1 | 1 | ✓ | Reflections only |
| C2 | 2 | ✗ | 180° rotations |
| D2 | 2 | ✓ | 180° rotations + reflections |
| C4 | 4 | ✗ | 90° rotations |
| D4 | 4 | ✓ | 90° rotations + reflections |
| C8 | 8 | ✗ | 45° rotations |
| D8 | 8 | ✓ | 45° rotations + reflections |

**Architecture difference from C4InvariantCNN:** Instead of `GroupPooling`, this model uses a 1×1 equivariant convolution projecting to trivial (invariant) representations before spatial pooling, giving more control over how invariance is achieved.

---

### `CnInvariantCNN`

A parameterized cyclic-group CNN where `n` controls the rotation order (C4, C8, C16, etc.). The linear layer input size scales with `n`: `64 * n` features after `GroupPooling`.

---

## Datasets

| Dataset | Description |
|---------|-------------|
| **MNIST** | Standard 28×28 handwritten digits, used for training |
| **Rotated MNIST** | Continuously rotated versions of MNIST (from U. Montreal), used for evaluation |
| **90°-rotated MNIST** | MNIST with deterministic 0°/90°/180°/270° rotations, used to test discrete invariance |

---

## Experiments

### Part 1 — C4 vs Baseline on Rotated MNIST

Both models trained on standard MNIST, evaluated on Rotated MNIST:

- **Baseline CNN**: ~28% accuracy on rotated test set (strong on clean MNIST, poor generalization)
- **C4-Invariant CNN**: Significantly higher accuracy on rotated MNIST due to built-in rotational symmetry

### Part 2 — Testing C4 Invariance

The C4 model is evaluated on MNIST rotated at 0°, 90°, 180°, and 270°. A truly C4-invariant network should produce **identical accuracy** at all four orientations, since these are exactly the symmetries it encodes.

### Part 3 — Comparing Symmetry Groups

Multiple group architectures (D1, C2, D2, C4, D4, C8, D8) trained with matching data augmentation and evaluated on Rotated MNIST. Larger/richer groups generally improve robustness to unseen rotations, at the cost of increased model complexity.

---

## Key Concepts

**Equivariance** — A network is equivariant to a transformation if rotating the input produces a corresponding rotation of the feature maps (rather than arbitrary changes).

**Invariance** — A network is invariant if the final prediction is unchanged regardless of the input rotation. Achieved here via group pooling or projection to trivial representations.

**Regular representation** — For a group of order `|G|`, the regular representation has dimension `|G|`. A feature map with `k` regular fields has `k × |G|` channels total.

**GroupPooling** — Averages (or max-pools) over the group dimension, collapsing equivariant features into invariant scalars.

---

## Dependencies

```
torch
torchvision
e2cnn==0.2.3
numpy
matplotlib
scipy
tensorflow  # for dataset loading only
tensorflow_datasets
```

Install e2cnn:
```bash
pip install e2cnn
```

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

## References

- Weiler, M. & Cesa, G. (2019). [General E(2)-Equivariant Steerable CNNs](https://arxiv.org/abs/1911.08251). NeurIPS 2019.
- Cohen, T. & Welling, M. (2016). [Group Equivariant Convolutional Networks](https://arxiv.org/abs/1602.07576). ICML 2016.
- [e2cnn library](https://github.com/QUVA-Lab/e2cnn) and [e2cnn experiments](https://github.com/QUVA-Lab/e2cnn_experiments)
- [Rotated MNIST dataset](http://www.iro.umontreal.ca/~lisa/icml2007data/)
