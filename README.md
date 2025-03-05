# Directional Quality Assessment for Nonlinear Dimensionality Reduction

## Overview

This master's thesis introduces a novel approach to evaluating the quality of dimensionality reduction techniques, addressing critical limitations in existing assessment methods. The research focuses on developing more intuitive and robust metrics for understanding how high-dimensional data is transformed into lower-dimensional representations.

## Key Contributions

### Motivation
Dimensionality reduction is crucial for visualizing and analyzing complex, high-dimensional datasets. However, existing quality assessment methods often:
- Rely on simplistic neighborhood-based approaches
- Fail to capture the directional complexity of data structures
- Are highly sensitive to local distortions

### Proposed Methods

#### 1. Path-Based R_NX(K) Metric
- Adapts the traditional neighborhood metric to analyze shortest paths
- Provides insights into local and global structure preservation
- Offers a more nuanced evaluation of dimensionality reduction techniques

#### 2. Edit Distance Method
- Compares path sequences in high-dimensional and low-dimensional spaces
- Quantifies structural preservation using minimal edit operations
- Reveals subtle differences in data representation

### Key Innovations
- Introduces a "web-like" approach to data structure analysis
- Captures more complex relationships than traditional linear neighborhood methods
- Allows identification of high and low-quality zones in embeddings

## Methodology

### Experimental Approach
- Evaluated methods on standard datasets:
  - MNIST (handwritten digits)
  - COIL-20 (3D object images)
- Analyzed multiple dimensionality reduction techniques:
  - PCA
  - t-SNE
  - UMAP
  - Isomap
  - MDS

### Computational Challenges
- Developed acceleration techniques:
  - Multi-layer convex hull approaches
  - Alpha shapes for improved coverage
  - Random sampling strategies to reduce computational complexity

## Key Findings
- Path-based metrics provide more intuitive quality assessment
- Different dimensionality reduction techniques show distinct structural preservation characteristics
- Random sampling can effectively approximate full metric computation

## Potential Applications
- Improved data visualization quality assessment
- Enhanced understanding of dimensionality reduction techniques
- More reliable interpretation of complex datasets in fields like:
  - Biology
  - Machine Learning
  - Exploratory Data Analysis

## Future Work
- Refine visualization techniques
- Optimize graph construction methods
- Explore geodesic distance integration

## Getting Started

### Dependencies
- Python
- NumPy
- SciPy
- NetworkX

### Installation
```bash
git clone https://github.com/yourusername/dimensionality-reduction-quality
cd dimensionality-reduction-quality
pip install -r requirements.txt
```

## Citation
Remacle, P. (2024). Directional Quality Assessment for Nonlinear Dimensionality Reduction. Master's Thesis, École polytechnique de Louvain.

## License
[Specify your license, e.g., MIT License]

## Contact
Pierre Remacle - [your email]

**Disclaimer**: This research represents an initial exploration into more nuanced dimensionality reduction quality assessment. Continued research and refinement are encouraged.
