# Equivariant CNNs for Rotated MNIST Classification

This project explores the use of **group-equivariant convolutional neural networks** (G-CNNs) for classifying handwritten digits under arbitrary rotations. Using the [`e2cnn`](https://github.com/QUVA-Lab/e2cnn) library, we implement and compare models with various symmetry groups — including cyclic (Cₙ) and dihedral (Dₙ) groups — against a standard baseline CNN on the rotated MNIST benchmark.

---

## Motivation

Standard CNNs learn independent filters for each orientation of a pattern, which is both parameter-inefficient and brittle when test images are rotated. Group-equivariant CNNs address this by enforcing symmetry constraints directly in the architecture: a pattern learned at one orientation is automatically shared across all orientations via weight-sharing. This makes them more data-efficient and significantly more robust to rotation.

---

## Results

### C4-Invariant Model vs. Baseline CNN

Both models were trained on a rotated MNIST dataset and evaluated on a held-out rotated test set:

| Model | Rotated MNIST Accuracy |
|---|---|
| Baseline CNN | 26.0% |
| C4-Invariant CNN | **96.3%** |

The C4 model achieves nearly 4× the accuracy of the baseline, demonstrating the power of baking rotational symmetry into the architecture.

The contrast in training behavior is stark — the baseline CNN fails to converge on rotated data, while the C4-invariant model converges smoothly and rapidly:

| Baseline CNN | C4-Invariant CNN |
|---|---|
| ![Baseline training loss](https://raw.githubusercontent.com/egeozgul/Steerable_CNNs/main/Figures/Figure1.png) | ![C4 training loss](https://raw.githubusercontent.com/egeozgul/Steerable_CNNs/main/Figures/Figure2.png) |

### C4 Model Accuracy by Rotation Angle

To probe where the C4 model struggles, it was evaluated separately on each 90° rotation of the standard MNIST test set:

| Rotation | Accuracy |
|---|---|
| 0° | 84.6% |
| 90° | 69.6% |
| 180° | 63.5% |
| 270° | 74.1% |

### Symmetry Group Comparison

A unified `EquivariantCNN` class was used to train and evaluate models across six different symmetry groups. Each model was evaluated on both a custom dataset matching its symmetry and the standard rotated MNIST benchmark:

| Model | Symmetry | Custom Dataset | Rotated MNIST |
|---|---|---|---|
| D1 | Reflections only | 94.26% | 94.00% |
| C2 | 180° rotations | 96.04% | 47.42% |
| D2 | 180° rotations + reflections | 94.54% | 94.81% |
| D4 | 90° rotations + reflections | 97.36% | **96.78%** |
| C8 | 45° rotations | 99.01% | 42.82% |
| D8 | 45° rotations + reflections | 97.01% | 97.24% |

**Key takeaway:** Models with reflection symmetry (Dₙ groups) generalize well to the rotated MNIST benchmark. Pure cyclic models (C2, C8) achieve high accuracy on their custom datasets but fail to generalize, suggesting that reflections are a crucial inductive bias for this task. D4 and D8 match or exceed the original C4 model on rotated MNIST.

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

## Project Structure

```
.
├── README.md
├── equivariant_cnn.py       # EquivariantCNN model definition (Cn and Dn)
├── data_utils.py            # prepare_mnist() — data loading and augmentation
├── train.py                 # Training loop
└── evaluate.py              # Evaluation scripts
```

---

## Model Architecture

The core model is `EquivariantCNN`, which accepts two parameters:
- `N` — the rotation order (e.g., 4 for 90° rotational symmetry)
- `reflections` — whether to include reflection symmetry (Dₙ vs. Cₙ)

It consists of four equivariant convolutional blocks (each with `R2Conv` → `InnerBatchNorm` → `ReLU`), followed by spatial average pooling, group pooling to achieve invariance, and two fully connected layers for classification.

```python
model = EquivariantCNN(N=4, reflections=True)   # D4
model = EquivariantCNN(N=4, reflections=False)  # C4
model = EquivariantCNN(N=8, reflections=True)   # D8
```

---

## Data Preparation

The `prepare_mnist` function generates training and evaluation dataloaders from the standard MNIST dataset, applying rotations and optional reflections to match the target symmetry group:

```python
train_loader, eval_loader = prepare_mnist(
    batch_size=64,
    rotation_order=4,
    include_reflections=True
)
```

---

## Dependencies

- Python 3.8+
- PyTorch
- [e2cnn](https://github.com/QUVA-Lab/e2cnn)
- torchvision

Install dependencies:

```bash
pip install torch torchvision e2cnn
```

---

## References

- Weiler & Cesa, *General E(2)-Equivariant Steerable CNNs*, NeurIPS 2019
- Cohen & Welling, *Group Equivariant Convolutional Networks*, ICML 2016
- [e2cnn library](https://github.com/QUVA-Lab/e2cnn)
