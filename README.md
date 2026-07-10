# Wavelet Scattering & TissueMNIST Classification

A medical image classifier deploying Wavelet Scattering Transforms (WST) for translation-invariant feature extraction on the TissueMNIST dataset. Implements PCA dimensionality reduction and a backpropagation-free analytical matrix pseudoinverse classifier (15x training speedup over SGD/Adam).

## Features
- Translation and rotation-invariant feature extraction using 2nd order WST.
- Dimensionality reduction via Principal Component Analysis (PCA).
- Backpropagation-free classification using an analytical matrix pseudoinverse solver, achieving 15x training acceleration.

## Tech Stack
- PyTorch
- Wavelet Scattering Transform (WST)
- PCA
- Linear Algebra
- Pseudoinverse Classifier

## Getting Started
To configure and run the project locally, clone the repository and execute the setup instructions:

```bash
git clone https://github.com/Raghuram-sekar/Wavelet-Scattering-TissueMNIST.git
cd Wavelet-Scattering-TissueMNIST

# Execute local setup commands:
# Open Jupyter notebooks in project root
jupyter notebook wavelet_scattering_tissuemnist.ipynb
```
