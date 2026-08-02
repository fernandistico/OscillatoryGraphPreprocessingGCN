# Oscillatory Graph Preprocessing: A Reusable, Amortizable Structural Prior for Deep GCNs

**Paper Status:** Under Review  
**Repository Version:** Pre-Publication (Superficial Documentation)

---

## Overview

This repository contains the official implementation of **"Oscillatory Graph Preprocessing: A Reusable, Amortizable Structural Prior for Deep GCNs,"** a method that applies fixed, untrained nonlinear dynamical systems as one-time graph preprocessing to stabilize deep Graph Convolutional Networks (GCNs) against oversmoothing.

### Main Contribution

We demonstrate that precomputing structural descriptors from graph-conditioned oscillatory dynamics—**Kuramoto**, **Stuart–Landau**, and **Wilson–Cowan**—produces reusable node embeddings that:

- **Preserve structural distinctions** in deep GCNs where repeated message passing normally erases them
- **Delay oversmoothing** with depth-dependent benefits (neutral to harmful at shallow layers, beneficial at $L \geq 6$)
- **Achieve statistically significant gains** on larger, noisier graphs (e.g., +23.3 pp on PubMed at $L=12$)
- **Require no learnable parameters** in the downstream model—computed once, reused across training runs
- **Maintain deterministic CPU/GPU equivalence** for reproducible deployment

---

## Authors

- **Fernando Vera Buschmann** — Dept. of Data Science, NJIT | [fv54@njit.edu](mailto:fv54@njit.edu)
- **Isidro Gauto** — Fed. Dept. of Biological Sciences, NJIT & Rutgers University | [ig242@njit.edu](mailto:ig242@njit.edu)
- **Dahlia Musa** — Dept. of Information Systems, NJIT | [dm79@njit.edu](mailto:dm79@njit.edu)
- **Horacio G. Rotstein** — Fed. Dept. of Biological Sciences, NJIT & Rutgers University | [horacio@njit.edu](mailto:horacio@njit.edu)
- **Vincent Oria** — Dept. of Computer Science, NJIT | [oria@njit.edu](mailto:oria@njit.edu)

---

## Key Findings (Summary)

| Aspect | Finding |
|--------|---------|
| **Depth Dependence** | Oscillatory descriptors are neutral or harmful at shallow GCNs ($L=1$–$2$) but become protective as oversmoothing develops ($L \geq 6$) |
| **Peak Performance** | +23.3 percentage points on PubMed at $L=12$ (z ≈ 7.4) with 9.5× reduction in across-seed variability |
| **Generalization** | Directionally consistent gains across Cora and CiteSeer, without comparable statistical significance |
| **Computational Cost** | Deterministic preprocessing computed once; CPU/GPU crossover near $N \approx 37,000$ nodes |
| **Reproducibility** | Bit-for-bit deterministic under fixed seeds; frozen-model CPU/GPU equivalence confirmed |

---

## Quick Start

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0.0
- PyTorch Geometric ≥ 2.3.0
- NumPy, scikit-learn, pandas, matplotlib, seaborn

See `requirements.txt` for full version specifications.

### Basic Usage

*(Detailed implementation is withheld until post-publication. The following is pseudocode.)*

```python
# 1. Load graph and compute oscillatory descriptors (one-time)
descriptors = oscillatory_preprocessing(
    adjacency_matrix=A,
    node_features=X,
    models=['kuramoto', 'stuart_landau', 'wilson_cowan']
)

# 2. Augment node features
X_augmented = np.concatenate([X, descriptors], axis=1)

# 3. Train standard GCN with augmented features
model = GCN(input_dim=X_augmented.shape[1], depth=L, hidden_dim=64)
train_gnn(model, X_augmented, A, train_indices)
```

---

## Datasets

Experiments use three standard citation benchmarks from the Planetoid collection:

| Dataset | Nodes | Edges | Features | Classes | Splits |
|---------|-------|-------|----------|---------|--------|
| Cora | 2,708 | 5,429 | 1,433 | 7 | Transductive (standard) |
| CiteSeer | 3,327 | 4,732 | 3,703 | 6 | Transductive (standard) |
| PubMed | 19,717 | 44,338 | 500 | 3 | Transductive (standard) |

All results follow leakage-free evaluation protocols: edge partitioning occurs **before** preprocessing.

---

## Repository Structure

```
OscillatoryGraphPreprocessingGCN/
├── Models/               # Oscillatory system implementations
│   ├── kuramoto.py       # Phase synchronization model
│   ├── stuart_landau.py  # Amplitude–phase dynamics
│   └── wilson_cowan.py   # Excitatory–inhibitory populations
├── Analysis/             # Evaluation and visualization
│   ├── metrics.py        # Accuracy, stability, CKA metrics
│   └── depth_sweep.py    # Depth robustness evaluation
├── Task/                 # Experimental orchestration
│   └── train_gnn.py      # Training loops and validation
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── PAPER.md              # Full paper reference and abstract
├── METHOD_OVERVIEW.md    # High-level method description
└── REPRODUCIBILITY.md    # General reproducibility guidelines
```

---

## Methodology (Conceptual Overview)

The approach consists of three phases:

### 1. **Oscillatory Preprocessing** (One-Time)
Simulate three fixed, untrained nonlinear dynamical systems on the graph:
- **Kuramoto**: Phase synchronization in coupled oscillators
- **Stuart–Landau**: Amplitude–phase dynamics near a Hopf bifurcation
- **Wilson–Cowan**: Excitatory–inhibitory neural population dynamics

Each system runs for a fixed integration horizon and produces node-level descriptors capturing network response properties.

### 2. **Feature Augmentation** (Deterministic)
Concatenate oscillatory descriptors with original node attributes. This augmentation is computed once and reused across all training runs and random seeds.

### 3. **GCN Training** (Standard)
Train a standard GCN on augmented features. The method does not require modifying the GCN architecture or training procedure.

**Full technical details** (equations, parameters, numerical methods) are provided in the paper.

---

## Evaluation Protocol

### Metrics

- **Node Classification:** Accuracy (%)
- **Link Prediction:** ROC-AUC, filtered Mean Reciprocal Rank (MRR), Hits@K
- **Stability:** Across-seed standard deviation
- **Representational Similarity:** Linear Centered Kernel Alignment (CKA)

### Leakage-Free Evaluation

All edge partitioning (train/validation/test) occurs **before** preprocessing. This ensures that preprocessing operates only on training edges and cannot leak information from validation or test sets.

### Reproducibility

- **Random Seeds:** 8 independent seeds per configuration
- **Deterministic GPU Execution:** Frozen-model numerical equivalence verified
- **No Hyperparameter Tuning:** Fixed parameters across all experiments (detailed in paper)

---

## Main Results

### Depth Robustness (Headline Finding)

The central observation is **strongly depth-dependent**:

- **Shallow regime** ($L=1$–$2$): Oscillatory augmentation is **neutral or harmful** (degradation up to 16 pp on PubMed)
- **Transition regime** ($L \approx 6$): Performance gap narrows
- **Deep regime** ($L \in [8, 12]$): Augmented models significantly outperform baselines
  - PubMed: **+23.3 pp at $L=12$** (statistically significant, $z \approx 7.4$)
  - Cora: **+5.9 pp at $L=10$** (directional, within noise)
  - CiteSeer: **+2.2 pp at $L=10$** (directional, within noise)
- **Extreme depth** ($L=16$): Both models approach degenerate performance

### Stability Improvement

On PubMed at $L=10$, across-seed standard deviation drops from **0.117** to **0.012**—a **9.5× reduction**. This indicates that oscillatory encodings stabilize optimization trajectories in the oversmoothing regime.

### Computational Efficiency

- **Preprocessing Cost:** Modest one-time cost; $O(|\mathcal{E}|)$ sparse complexity
- **CPU vs. GPU:** CPU-efficient for small graphs; GPU gains speedup ($\sim 2×$) near $N \approx 37,000$ nodes
- **Amortization Argument:** Cost is independent of training runs, seeds, and architectures

---

## Current Status & Confidentiality

### **Implementation Confidentiality**

This repository provides:
- Paper abstract, results, and key findings
- General methodology overview and conceptual description
- Reproducibility protocol and dataset references
- Requirements and quick-start guide

This repository **does not yet include**:
- Exact numerical parameters (Kuramoto $\omega_0$, Stuart–Landau $\alpha_0$, Wilson–Cowan weights, etc.)
- RK4 integration details (step size, burn-in duration, feature recording window)
- Descriptor computation code (PLV computation, standardization procedures, etc.)
- Full training scripts and hyperparameter configurations

**Full implementation details will be released upon publication.** This is standard practice in competitive research to prevent simultaneous circumvention of novel contributions.

---

## Citation

```bibtex
@article{vera2026oscillatory,
  title   = {Oscillatory Graph Preprocessing: A Reusable, Amortizable Structural Prior for Deep {GCNs}},
  author  = {Vera Buschmann, Fernando and Gauto, Isidro and Musa, Dahlia and Rotstein, Horacio G. and Oria, Vincent},
  year    = {2026},
  status  = {Under Review},
  journal = {(venue TBD)},
  note    = {Available at \url{https://github.com/fernandistico/OscillatoryGraphPreprocessingGCN}}
}
```

---

## Contact & Support

For questions about the paper or to report issues:

- **Email:** Fernando Vera Buschmann ([fv54@njit.edu](mailto:fv54@njit.edu))

---

## License

MIT License © 2026 Fernando Vera Buschmann

See LICENSE file for full terms.

---

## Acknowledgments

- **Funding:** IG is supported by the Mathematical and Computational Biosciences Collective (MCBC) through the NJIT Collaborative Research and Innovation Strategic Partnership (CRISP) Investment Plan.
- **HPC Resources:** Portions of this work were performed on HPC resources (Wulver cluster at NJIT).
- **Tooling:** Authors used ChatGPT and Claude for language editing. All content was reviewed and edited by authors, who take full responsibility for the final work.

---

## Related Work

Key references relevant to this research:

- **Oversmoothing in GNNs:** Oono & Suzuki (2020), Rusch et al. (2023)
- **Trainable Oscillator GNNs:** Nguyen et al. (2024), Zhang et al. (2026)
- **Structural Encodings:** Dwivedi et al. (2023), Rampášek et al. (2022)
- **Coupled Oscillators:** Pikovsky et al. (2001), Hoppensteadt & Izhikevich (1997)

Full bibliography available in the paper.

---

**Last Updated:** August 2, 2026 
