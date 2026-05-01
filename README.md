![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?style=flat&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

# Lego Brick Shape and Color Classification with PyTorch

A from-scratch PyTorch project that classifies **45 Lego brick shapes** and **9 color categories**
through a self-built dataset and a progressive series of three CNN architectures —
from a plain baseline to a lightweight Mini-ResNet —
designed as a structured ablation study into the roles of Batch Normalization and residual connections.

## Overview

|                          |                                                              |
| ------------------------ | ------------------------------------------------------------ |
| **Tasks**                | Shape classification (45 classes) + Color classification (9 classes), trained as two independent models |
| **Models**               | V1 · Plain CNN → V2 · CNN + BatchNorm → V3 · Mini-ResNet (progressive ablation) |
| **Dataset**              | Self-collected from official Lego 3D render pages (~367 source images across 45 shape classes); color variants generated via HSV hue shift and manual capture; offline augmentation produced 13,212 shape images and 10,008 color images; shape images were then split 5:1 into a training set (11,010) and a test set (2,382), with online augmentation applied to the training set only |
| **Training**             | Adam + CosineAnnealingLR + gradient clipping · 50 epochs (shape) / 30 epochs (color) |
| **Hardware**             | NVIDIA RTX 4090 / 5090 via AutoDL cloud · local development on Windows (Anaconda) |
| **Language / Framework** | Python 3.10 · PyTorch 2.5.1 · torchvision · CUDA 12.4        |
| **Evaluation**           | 2,382-image shape test set · 14,824-image color test set · per-class accuracy · confusion matrix |
| **Best Results**         | Shape: 93.37% (V3) · Color: 100.00% (V3)                     |

## Key Features

**Self-built dataset with a two-stage augmentation pipeline.**
All source images were manually collected from official Lego 3D render pages and organized into 45 shape classes. Rather than relying on a single augmentation pass, the pipeline is deliberately split into two stages: an offline stage that handles strong geometric transformations — scaling, positional shift, affine shear, and rotation — all performed on RGBA  transparent images before background compositing to eliminate edge-fill artifacts; and a lightweight online stage applied only during training for mild perturbation. This separation ensures that the test set remains clean and that the two stages do not compound into destructive distortion.

**Progressive ablation across three architectures.**
The three models are not arbitrary choices but form a controlled experiment: V1 establishes aplain CNN baseline with no normalization, V2 adds Batch Normalization and increased depth to isolate the effect of internal covariate shift correction, and V3 introduces residual connections to assess whether skip connections improve parameter efficiency beyond what BN alone provides. Each step changes exactly one structural variable, making the performance differences directly interpretable.

**Honest evaluation design with deliberate test set construction.**
Beyond the standard 5:1 split, a supplementary subset of images featuring less common oblique and near-overhead viewpoints was collected from the same render source and incorporated into the final test set. This was a conscious decision to stress-test model robustness rather than report inflated accuracy on familiar perspectives. The evaluation section also explicitly acknowledges the train-test correlation introduced by augmented-image-level splitting, rather
than treating the reported numbers as a measure of true generalization.

**Full engineering pipeline from data to visualization.**
The project covers every stage of a practical deep learning workflow: raw data collection, offline augmentation scripting, dataset splitting, online augmentation via a custom dataloader, model training with scheduler and gradient clipping, quantitative evaluation with confusion matrices and per-class breakdowns, multi-model comparison plots, architecture visualization, and a joint shape-plus-color prediction demo. All stages are scripted and reproducible.

## Dataset

### Shape Dataset

Original images were manually captured from the official Lego 3D render pages in Bright Red (PNG format), covering 45 brick and plate categories at three standard oblique viewing angles. L-shaped corner pieces (shape_08, shape_29) were photographed at 6 angles (3 views × 180°rotation) to capture both orientations; single-side stud pieces (shape_18–20, shape_41) were captured with the stud face explicitly visible. After merging the initial source images with a later batch of test-phase renders, the consolidated pool reached **~367 source images** (7–9 per class).

**Offline augmentation** (`data_augment_shape_v2.py`) — all transforms applied on RGBA before background compositing:

| Parameter            | Options            | Sampled     |
| -------------------- | ------------------ | ----------- |
| Background color     | 8 light pastels    | 2           |
| Scale                | 0.65 / 0.80 / 0.95 | 1           |
| Position             | center / 4 corners | 3           |
| Horizontal flip      | —                  | ×1          |
| Affine shear         | ≤ 5°, shift ≤ 3%   | ×2 variants |
| Rotation             | ± 15°              | ×3 variants |
| **Per-image output** | **2×1×3×1×2×3**    | **= 36**    |

→ ~367 × 36 = **13,212 images** total  
→ Split 5:1 (seed 42): **11,010 training / 2,202 test**  
→ A supplementary subset of challenging-viewpoint renders was augmented separately
(4× each) and merged into the test set → **final test set: 2,382 images**

**Online augmentation** (training set only, `utils_shape/dataset.py`):

```python
RandomCrop(224, padding=10)
RandomHorizontalFlip(p=0.5)
ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10)
GaussianNoise(max_std=5)
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

`AUGMENT_MULTIPLY = 10` → ~110,100 effective samples per epoch.
`RandomRotation` excluded — offline stage already covers ±15°.

---

### Color Dataset

Color images share the same 45-class brick geometry as the shape dataset. Six colors (Red through Rose) were generated from the Bright Red source images via HSV hue shift; Gray via desaturation; White and Black from separate manual captures.

| ID        | Color  | Method         |
| --------- | ------ | -------------- |
| colour_00 | Red    | Original (0°)  |
| colour_01 | Yellow | Hue +60°       |
| colour_02 | Green  | Hue +120°      |
| colour_03 | Cyan   | Hue +180°      |
| colour_04 | Blue   | Hue +240°      |
| colour_05 | Rose   | Hue +300°      |
| colour_06 | Gray   | Desaturation   |
| colour_07 | White  | Manual capture |
| colour_08 | Black  | Manual capture |

Offline augmentation produced **10,008 images** (9 classes × 139 source images × 8 backgrounds). The test set contains **14,824 images**: 1,800 per hue-shifted class (colour_00–06) and 1,112 each for White and Black.

---

### A Note on Train-Test Correlation

The 5:1 split was performed at the augmented-image level rather than the source-image level, meaning augmented variants of the same source render may appear in both the training and test sets. This introduces a degree of correlation that likely inflates the reported accuracy figures. Results should be read as in-distribution performance rather than true generalization to unseen images. This limitation is discussed further in [Limitations & Future Work](#limitations--future-work).

## Model Architecture

Two independent sets of models were designed for the shape and color tasks respectively. Within each set, the three architectures form a controlled ablation: one variable changes at each step, making the performance differences directly attributable to a specific structural component.

### Shape Classification Models (45 classes)

| Model  | Architecture                                                 | BatchNorm | Residual | Parameters |   LR   |
| ------ | ------------------------------------------------------------ | :-------: | :------: | :--------: | :----: |
| **V1** | 8-layer CNN · 4 blocks (64→128→256→512) · GlobalAvgPool + 2× FC |     ✗     |    ✗     |   ~15 M    | 0.0002 |
| **V2** | 12-layer CNN · 5 blocks (64→128→256→512→512) · BN after every conv |     ✓     |    ✗     |   ~22 M    | 0.001  |
| **V3** | 17-layer Mini-ResNet · Stem + 4 residual layers (64→128→256→512) · 1×1 projection shortcut |     ✓     |    ✓     |   ~11 M    | 0.001  |

```
V1  [Conv→ReLU→Pool] ×4 → GlobalAvgPool → FC → FC → Softmax
V2  [Conv→BN→ReLU→Pool] ×5 → GlobalAvgPool → FC → FC → Softmax
V3  Stem → [ResBlock(Conv→BN→ReLU→Conv→BN) + shortcut] ×4 → GlobalAvgPool → FC → Softmax
```

### Color Classification Models (9 classes)

Color recognition is a simpler task by nature — the discriminative signal is global and low-frequency — so all three color models are kept substantially smaller than their shape counterparts while preserving the same ablation structure.

| Model         | Architecture                                               | BatchNorm | Residual | Parameters |
| ------------- | ---------------------------------------------------------- | :-------: | :------: | :--------: |
| **Colour V1** | 4-layer CNN · 2 blocks (32→64) · GlobalAvgPool + 2× FC     |     ✗     |    ✗     |   ~45 K    |
| **Colour V2** | 6-layer CNN · 3 blocks (32→64→128) · BN after every conv   |     ✓     |    ✗     |   ~120 K   |
| **Colour V3** | 9-layer Mini-ResNet · Stem + 3 residual layers (32→64→128) |     ✓     |    ✓     |   ~80 K    |

### Architecture Diagrams

<p align="center">
  <img src="model_structures/shape_v1_architecture.png" width="30%" alt="Shape V1"/>
  <img src="model_structures/shape_v2_architecture.png" width="30%" alt="Shape V2"/>
  <img src="model_structures/shape_v3_architecture.png" width="30%" alt="Shape V3"/>
</p>
<p align="center"><i>Shape models: V1 (Plain CNN) · V2 (BN-CNN) · V3 (Mini-ResNet)</i></p>

<p align="center">
  <img src="model_structures/colour_v1_architecture.png" width="30%" alt="Colour V1"/>
  <img src="model_structures/colour_v2_architecture.png" width="30%" alt="Colour V2"/>
  <img src="model_structures/colour_v3_architecture.png" width="30%" alt="Colour V3"/>
</p>
<p align="center"><i>Color models: V1 (Plain CNN) · V2 (BN-CNN) · V3 (Mini-ResNet)</i></p>

## Training Configuration

### Common Setup

| Setting                 | Shape Task                                   | Color Task                                   |
| ----------------------- | -------------------------------------------- | -------------------------------------------- |
| Optimizer               | Adam                                         | Adam                                         |
| Loss Function           | CrossEntropyLoss                             | CrossEntropyLoss                             |
| LR Scheduler            | CosineAnnealingLR (T_max = 50, η_min = 1e-6) | CosineAnnealingLR (T_max = 30, η_min = 1e-6) |
| Epochs                  | 50                                           | 30                                           |
| Batch Size              | 64                                           | 64                                           |
| Input Size              | 224 × 224                                    | 224 × 224                                    |
| Gradient Clipping       | max_norm = 2.0                               | max_norm = 1.0                               |
| Learning Rate (V2 / V3) | 0.001                                        | 0.001                                        |
| Learning Rate (V1)      | 0.0002                                       | 0.0002                                       |

### Why V1 Requires a Lower Learning Rate

V1 contains no Batch Normalization. Without BN's internal covariate shift correction, activation distributions across layers can drift sharply under the combined effect of strong augmentation and a standard learning rate, causing gradient explosion early in training — in practice, V1 initially converged to ~3.6% accuracy (near random chance for 45 classes) at lr = 0.001. Two adjustments were made to stabilize it:

- **Learning rate reduced to 0.0002** — smaller parameter updates prevent runaway activation drift in the early epochs.
- **Gradient clipping (max_norm = 2.0 for shape, 1.0 for color)** — applied after`loss.backward()` to hard-cap gradient norms before the optimizer step.

V2 and V3 both include BN and converge stably at the standard rate of 0.001. This difference in required learning rate directly reflects the practical cost of removing Batch Normalization: without it, even a well-designed CNN becomes
significantly harder to train under strong data augmentation.

## Results & Visualization

### Shape Classification (45 classes · 2,382-image test set)

| Model            | Parameters | Test Accuracy | Avg Inference Time |
| ---------------- | :--------: | :-----------: | :----------------: |
| V1 · Plain CNN   |   ~15 M    |    92.74%     |      0.73 ms       |
| V2 · CNN + BN    |   ~22 M    |    93.07%     |      1.09 ms       |
| V3 · Mini-ResNet |   ~11 M    |  **93.37%**   |    **0.27 ms**     |

V3 achieves the highest accuracy with the fewest parameters and the lowest inference latency — outperforming V2 on all three metrics despite being nearly half its size.

<p align="center">
  <img src="results_shape_compare/model_compare_summary.png" width="85%"
       alt="Shape Three-Model Comparison"/>
</p>


<details>
<summary>Training curves · Confusion matrix · Per-class accuracy (click to expand)</summary>
<br>
<p align="center">
  <img src="results_shape_train/model_v1_loss.png"    width="30%" alt="V1 Loss"/>
  <img src="results_shape_train/model_v2_loss.png"    width="30%" alt="V2 Loss"/>
  <img src="results_shape_train/model_v3_loss.png"    width="30%" alt="V3 Loss"/>
</p>
<p align="center"><i>Training loss curves — V1 · V2 · V3</i></p>


<p align="center">
  <img src="results_shape_test/model_v1_confusion_matrix.png" width="30%" alt="V1 Confusion"/>
  <img src="results_shape_test/model_v2_confusion_matrix.png" width="30%" alt="V2 Confusion"/>
  <img src="results_shape_test/model_v3_confusion_matrix.png" width="30%" alt="V3 Confusion"/>
</p>
<p align="center"><i>Confusion matrices — V1 · V2 · V3</i></p>

<p align="center">
  <img src="results_shape_test/model_v1_per_class_acc.png" width="30%" alt="V1 Per-class"/>
  <img src="results_shape_test/model_v2_per_class_acc.png" width="30%" alt="V2 Per-class"/>
  <img src="results_shape_test/model_v3_per_class_acc.png" width="30%" alt="V3 Per-class"/>
</p>
<p align="center"><i>Per-class accuracy — V1 · V2 · V3</i></p>

</details>

---

### Color Classification (9 classes · 14,824-image test set)

| Model                   | Parameters | Test Accuracy | Avg Inference Time |
| ----------------------- | :--------: | :-----------: | :----------------: |
| Colour V1 · Plain CNN   |   ~45 K    |    99.92%     |      0.26 ms       |
| Colour V2 · CNN + BN    |   ~120 K   |    99.99%     |      0.23 ms       |
| Colour V3 · Mini-ResNet |   ~80 K    |  **100.00%**  |    **0.10 ms**     |

All three color models perform near-perfectly, confirming that color discrimination from consistent render-style images is a well-constrained task at this scale. V3 again achieves the best accuracy with the fastest inference speed.

<p align="center">
  <img src="results_colour_compare/colour_compare_summary.png" width="85%"
       alt="Color Three-Model Comparison"/>
</p>


<details>
<summary>Training curves · Confusion matrix · Per-class accuracy (click to expand)</summary>
<br>
<p align="center">
  <img src="results_colour_train/colour_v1_loss.png" width="30%" alt="Colour V1 Loss"/>
  <img src="results_colour_train/colour_v2_loss.png" width="30%" alt="Colour V2 Loss"/>
  <img src="results_colour_train/colour_v3_loss.png" width="30%" alt="Colour V3 Loss"/>
</p>
<p align="center"><i>Training loss curves — Colour V1 · V2 · V3</i></p>


<p align="center">
  <img src="results_colour_test/colour_v1_confusion_matrix.png" width="30%" alt="Colour V1 Confusion"/>
  <img src="results_colour_test/colour_v2_confusion_matrix.png" width="30%" alt="Colour V2 Confusion"/>
  <img src="results_colour_test/colour_v3_confusion_matrix.png" width="30%" alt="Colour V3 Confusion"/>
</p>
<p align="center"><i>Confusion matrices — Colour V1 · V2 · V3</i></p>

<p align="center">
  <img src="results_colour_test/colour_v1_per_class_acc.png" width="30%" alt="Colour V1 Per-class"/>
  <img src="results_colour_test/colour_v2_per_class_acc.png" width="30%" alt="Colour V2 Per-class"/>
  <img src="results_colour_test/colour_v3_per_class_acc.png" width="30%" alt="Colour V3 Per-class"/>
</p>
<p align="center"><i>Per-class accuracy — Colour V1 · V2 · V3</i></p>

</details>

## Analysis & Discussion

***\*Batch Normalization is essential for training stability under strong augmentation.\****

Without BN, V1 initially failed to learn entirely — converging to ~3.6% accuracy, barely above random chance for a 45-class problem. The root cause is that strong augmentation causes large shifts in the input distribution seen by each layer, and without normalization these shifts accumulate across depth, destabilizing gradient flow. Reducing the learning rate and adding gradient clipping brought V1 to stable convergence, but the need for these workarounds itself demonstrates how much BN simplifies the optimization landscape.

**Residual connections improve parameter efficiency, not just accuracy.** 

V3 scores the highest accuracy (93.37%) on shape classification while using the fewest parameters (~11 M) and running nearly 4× faster than V2 (0.27 ms vs. 1.09 ms). This suggests that skip connections do more than prevent vanishing gradients — they allow the network to reuse features across layers more effectively, so fewer parameters are needed to reach the same representational capacity. Adding more layers and parameters, as in V2, does not automatically lead to better results without a mechanism for  efficient gradient flow.

**The two-stage augmentation design requires deliberate coordination.** 

An earlier version of the pipeline applied aggressive rotation (±30°) both offline and online, which caused two problems: gray corner fill artifacts from repeated rotation on already-composited images, and over-augmentation that blurred the decision boundary. The fix was to assign distinct roles to each stage — offline handles all strong geometric diversity, online adds only mild perturbation. This coordination between the two stages was a key factor in reaching stable, consistent training.

**Shape and color present fundamentally different levels of difficulty.** 

All three color models exceeded 99.9% accuracy on a 14,824-image test set, while shape models plateaued in the 92–93% range despite using far more parameters and training data. The gap reflects the nature of the two tasks: color is a global, low-frequency signal that even a small network can reliably capture, while distinguishing 45 geometrically similar brick shapes — many differing only in stud count or aspect ratio — requires the model to learn fine-grained spatial features that are sensitive to viewpoint and scale variation.

## Limitations & Future Work

### Current Limitations

**Train-test correlation from augmented-image-level splitting.** 

The 5:1 split was applied after offline augmentation, so different augmented variants of the same source image can appear in both the training and test sets. This means the test set is not fully independent, and the reported accuracies reflect how well the models learned the augmented data distribution rather than true generalization to unseen images. A stricter evaluation would split at the source-image level, ensuring no render appears in both sets.

**Limited generalization beyond the training domain.**
All images — both training and test — originate from the same official Lego 3D render source, sharing a consistent lighting style, background palette, and viewpoint range. When tested against real-world photographs of Lego bricks, model performance drops sharply, primarily because side-facing and bottom-facing angles are entirely absent from
the training data. The models are therefore best understood as render-domain classifiers rather than general-purpose brick detectors.

**Fixed and narrow viewpoint coverage.**
Source images cover three standard oblique angles. Classes with asymmetric features — such as single-side stud bricks (shape_18–20, shape_41) and corner pieces (shape_08, shape_29) — are inherently more sensitive to viewpoint change, and this is reflected in their relatively lower per-class accuracy in the test results.

### Future Work

**Source-image-level train-test split.**
Rebuilding the split at the source-image level would produce a genuinely independent test set and give a more honest picture of model generalization.

**Viewpoint and domain expansion.**
Adding renders from a wider range of angles — including side-facing and bottom-facing views — would directly address the two main failure modes identified above. Incorporating real-photo data, even in small quantities, could further close the render-to-real gap.

**Joint shape and color inference.**
The two models currently run as separate pipelines. A natural next step would be to combine them into a single end-to-end system that outputs both shape and color labels from one forward pass, which would also allow shared feature extraction in early layers.

## Project Structure

```
Lego-bricks-classification-based-on-CNN/
├── raw_data_shape/                  # ~367 source renders (45 classes, Bright Red)
├── raw_data_colour/                 # 9-class color source images
├── raw_data_test/
│   └── add/             # Challenging-viewpoint renders (supplementary test source)
│   └── rad/               # rad pictures used for shape test         
│   └── black/             # 136 black pictures used for colour test
│   └── white/             # 136 white pictures used for colour test
│
├── layer_one_shape/                 # Offline-augmented shape images (13,212)
├── layer_one_colour/                # Offline-augmented color images (10,008)
├── train_data_shape/                # Shape training split (11,010)
│
├── models_shape/                    # Shape model definitions
│   ├── model_v1.py                  # Plain CNN (~15 M params)
│   ├── model_v2.py                  # CNN + BN (~22 M params)
│   └── model_v3.py                  # Mini-ResNet (~11 M params)
├── models_colour/                   # Color model definitions
│   ├── model_v1_colour.py           # Plain CNN (~45 K params)
│   ├── model_v2_colour.py           # CNN + BN (~120 K params)
│   └── model_v3_colour.py           # Mini-ResNet (~80 K params)
│
├── utils_shape/
│   └── dataset.py                   # Shape dataloader + online augmentation
├── utils_colour/
│   └── dataset_colour.py            # Color dataloader + online augmentation
│   └── data augment colour layer1.py# offline augmentation for raw_data_colour
│   └── add white black.py           # add white and black pictures to layer_one_colour
│
├── train/
│   ├── train_shape.py               # Shape training script  (--model v1/v2/v3)
│   └── train_colour.py              # Color training script  (--model v1/v2/v3)
├── test/
│   ├── test_shape.py                # Shape evaluation script
│   ├── test_colour.py               # Color evaluation script
│   ├── test_dataset_shape/          # Shape test set (2,382 images)
│   └── test_dataset_colour/         # Color test set (14,824 images)
├── compare/
│   ├── compare_models.py            # Three-model comparison for shape
│   └── compare_colour.py            # Three-model comparison for color
│
├── checkpoints_shape/               # Saved shape model weights (.pth)
├── checkpoints_colour/              # Saved color model weights (.pth)
│
├── results_shape_train/             # Shape training curves (loss / acc / lr)
├── results_shape_test/     # Shape test outputs (confusion matrix / per-class acc / errors)
├── results_shape_compare/           # Shape comparison plots
├── results_colour_train/            # Color training curves (loss / acc / lr)
├── results_colour_test/    # Color test outputs (confusion matrix / per-class acc / errors)
├── results_colour_compare/          # Color comparison plots
│
├── model_structures/                # Architecture diagrams (6 models)
│
├── data_augment_shape_v2.py         # Offline augmentation script (shape)
├── data_augment_shape_add.py        # Augmentation for challenging-viewpoint images
├── split_dataset_shape.py           # Train/test split script
├── visualize_models.py              # Architecture diagram generator
├── predict.py                       # Joint shape + color prediction demo
├── input_images                     # Data source for predict.py:you can put photos here and predict their shape and colour using predict.py
├── lego_mapping.csv                 # Class index ↔ Lego part ID mapping
├── requirements.txt
└── README.md

```

## Quick Start

### Installation

```bash
git clone https://github.com/mathewdaron/Lego-bricks-classification-based-on-CNN.git
cd Lego-bricks-classification-based-on-CNN
pip install -r requirements.txt
```

> For GPU training, install the CUDA-compatible PyTorch build from the [official PyTorch website](https://pytorch.org/get-started/locally/) before running the above.

### Train

```bash
# Shape classification — choose model with --model v1 / v2 / v3
python train/train_shape.py --model v3

# Color classification
python train/train_colour.py --model v3
```

Checkpoints are saved to `checkpoints_shape/` and `checkpoints_colour/`.  
Training curves are saved to `results_shape_train/` and `results_colour_train/`.

### Test

```bash
# Shape classification
python test/test_shape.py --model v3

# Color classification
python test/test_colour.py --model v3
```

Results — overall accuracy, confusion matrix, per-class breakdown, and misclassified
samples — are saved to `results_shape_test/` and `results_colour_test/`.

### Compare Models

```bash
# Shape — compares all three models from saved test results
python compare/compare_models.py

# Color
python compare/compare_colour.py
```

Comparison plots are saved to `results_shape_compare/` and `results_colour_compare/`.

### Predict

```bash
# Joint shape + color prediction on a single image
python predict.py --input_images path/to/your/image.png
```

---

## License

This project is released under the [MIT License](LICENSE).
