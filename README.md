# Wavelet Scattering & TissueMNIST Classification
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=SciPy&logoColor=white)

## Overview
A medical image classifier deploying Wavelet Scattering Transforms (WST) for translation-invariant feature extraction on the TissueMNIST dataset. Implements PCA dimensionality reduction and a backpropagation-free analytical matrix pseudoinverse classifier (15x training speedup over SGD/Adam).

## System Architecture
```\n[Relational Database / Core API Architecture]\n```

## Features
- Translation and rotation-invariant feature extraction using 2nd order WST.
- Dimensionality reduction via Principal Component Analysis (PCA).
- Backpropagation-free classification using an analytical matrix pseudoinverse solver, achieving 15x training acceleration.

## Tech Stack
- PyTorch for dataset loading and tensor math
- Wavelet Scattering Transform for spatial-invariant representations
- Moore-Penrose pseudoinverse solvers for analytical classification weight calculation

## Getting Started
To configure and run the project locally, clone the repository and execute the setup instructions:

```bash
git clone https://github.com/Raghuram-sekar/Wavelet-Scattering-TissueMNIST.git
cd Wavelet-Scattering-TissueMNIST

# Execute local setup commands:
# Moore-Penrose Pseudoinverse Weight Calculation:
# W = Y * pinv(X)

jupyter notebook wavelet_scattering_tissuemnist.ipynb
```
