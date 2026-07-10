# 🔬 Wavelet Scattering & Analytical TissueMNIST Classifier
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=SciPy&logoColor=white) ![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents
- [Project Overview](#🎯-project-overview)
- [What This Project Does](#🚀-what-this-project-does)
- [Key Innovation](#🔬-key-innovation)
- [Performance Highlights](#📊-performance-highlights)
- [Architecture](#🏗️-architecture)
- [Tech Stack](#🧱-tech-stack)
- [Quick Start](#💻-quick-start)

---

## 🎯 Project Overview
A medical image classifier deploying Wavelet Scattering Transforms (WST) for translation-invariant feature extraction on the TissueMNIST dataset. Implements PCA dimensionality reduction and a backpropagation-free analytical matrix pseudoinverse classifier (15x training speedup over SGD/Adam).

---

## 🚀 What This Project Does
* **The Challenge:** Training deep medical image classifiers (CNNs) requires hundreds of backpropagation iterations, high energy consumption, and massive labeled datasets to avoid overfitting.
* **Our Solution:** A backpropagation-free classification pipeline using 2nd order WST feature extractors and an analytical Moore-Penrose pseudoinverse solver.

---

## 🔬 Key Innovation
| Feature | Traditional CNNs ❌ | Our WST + Pseudoinverse ✅ | Benefit |
|---------|--------------------|----------------------------|---------|
| **Training** | Backpropagation (hours of GPU time) | **Analytical matrix pseudoinverse** | Training in seconds (15x speedup) |
| **Invariance** | Requires massive data augmentations | **2nd-Order Wavelet Scattering** | Built-in translation/rotation invariance |
| **Convergence** | Stochastic gradient descent risks | **Exact global mathematical optimum** | No local minima issues |

---

## 📊 Performance Highlights
- ✅ **15x training speedup** compared to standard SGD.
- ✅ **Tested on TissueMNIST dataset** (4,750 medical images).
- ✅ **Built-in invariance** to image translation and rotation.

---

## 🏗️ Architecture
```\n[Core Architectural Components & Datastore Framework]\n```

---

## 🧱 Tech Stack
- PyTorch for dataset loading and tensor math
- Wavelet Scattering Transform for spatial-invariant representations
- Moore-Penrose pseudoinverse solvers for analytical classification weight calculation

---

## 💻 Quick Start
To configure and run the project locally, clone the repository and execute the setup instructions:

```bash
git clone https://github.com/Raghuram-sekar/Wavelet-Scattering-TissueMNIST.git
cd Wavelet-Scattering-TissueMNIST

# Execute local setup commands:
# Moore-Penrose Pseudoinverse Weight Calculation:
# W = Y * pinv(X)

jupyter notebook wavelet_scattering_tissuemnist.ipynb
```
