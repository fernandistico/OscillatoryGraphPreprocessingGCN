## Article under review

# Oscillatory Graph Preprocessing: A Reusable, Amortizable Structural Prior for Deep GCNs

Official implementation of the paper **"Oscillatory Graph Preprocessing: A Reusable, Amortizable Structural Prior for Deep GCNs."**

This repository implements reusable oscillatory graph descriptors based on **Kuramoto**, **Stuart–Landau**, and **Wilson–Cowan** dynamics as a one-time graph preprocessing stage for Graph Convolutional Networks (GCNs). The proposed approach introduces structural priors that improve the robustness of deep GCNs against oversmoothing without adding trainable parameters to the downstream model.

## Authors

- **Fernando Vera Buschmann** — Department of Data Science, New Jersey Institute of Technology (NJIT)
- **Isidro Gauto** — Federated Department of Biological Sciences, NJIT & Rutgers University
- **Dahlia Musa** — Department of Information Systems, New Jersey Institute of Technology (NJIT)
- **Horacio G. Rotstein** — Federated Department of Biological Sciences, NJIT & Rutgers University
- **Vincent Oria** — Department of Computer Science, New Jersey Institute of Technology (NJIT

## Quick Start
#  dependencies
pip install torch torch_geometric numpy scikit-learn

## Key Findings

### RQ1: Minimal architecture (T=1)
On homophilous citation graphs, oscillators do NOT beat GCN within noise.
This is EXPECTED: low-pass filters are already near-optimal.

### RQ2: Integration horizon (T sweep)
- **SL**: maintains Dirichlet energy, does not oversmooth
- **Kuramoto**: shows cluster synchronisation (R_class → 1, R_global → 0)
- **WC**: oscillates weakly on citation networks; stronger on Chameleon

### RQ3: Integrator stability
- Euler diverges on SL (stiffness confirmed)
- RK4 and IMEX both stable; RK4 is cheaper
- IMEX is not strictly necessary (RK4 suffices)

### RQ4: Phase dynamics
Class-wise synchronisation (R_class ≈ 0.7–0.9) observed in SL/Kuramoto,
indicating the models form phase-aligned clusters rather than global sync.

### RQ5: Parameter matching
When matched for parameter count (GCN at d=128 vs SL at d=64), 
the gains disappear on homophilous graphs. On Chameleon/Squirrel,
WC shows modest but reproducible improvement (Δacc ≈ +0.8%).

# Hyperparameter Search Strategy

## SL Parameters (search space from paper)

- α ∈ [-0.5, 0.5]   : Hopf parameter (>0 → limit cycle)
- β ∈ [0.0, 2.0]    : cubic saturation
- ω ∈ [0.5, 2.0]    : natural frequency
- γ ∈ [-1.0, 2.0]   : shear parameter

## Kuramoto Parameters

- ω ∈ [0.5, 2.0]    : natural frequency (per channel)
- κ ∈ [0.5, 2.0]    : coupling strength

## Wilson-Cowan Parameters

Use classic Wilson & Cowan (1972) constants:
- c_ee, c_ei, c_ie, c_ii = (16, 12, 15, 3)
- P (external drive) ∈ [1.0, 1.5]  [CRITICAL: controls Hopf regime]
- k_E, k_I (graph couplings) ∈ [0.25, 1.0]

**Warning**: Weak initialization (c ~ 1, P = 0) never oscillates.
Always use oscAmp diagnostic to verify.


## Reproducibility

- All 10 random seeds logged
- Fixed PyTorch/PyG versions in `requirements.txt`
- Deterministic training (set_seed() before each run)
- Early stopping, gradient clipping, hyperparameter ranges documented
- Results CSV exported for external analysis


## License
MIT License

Copyright (c) 2026 Fernando Vera Buschmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Known Limitations

1. **Homophilous graphs**: oscillators provide no benefit (as predicted by theory)
2. **Heterophilous graphs**: WC shows modest gains, but requires careful initialization
3. **Scalability**: KM and WC have gather-scatter bottlenecks; not efficient >100k nodes
4. **Wilson-Cowan silent collapse**: oscAmp diagnostic is essential; weak params → fixed point

## License

MIT License. 

## Contact

For questions or issues, open a GitHub issue or email fv54 at NJIT. edu.

## References

- Zhang et al. (2026). "Stuart-Landau Oscillatory Graph Neural Network." *WWW'26*. arXiv:2511.08094
- Rusch et al. (2022). "Graph-Coupled Oscillator Networks." *ICML*. arXiv:2202.02296
- Nguyen et al. (2024). "From Coupled Oscillators to Graph Neural Networks." *AISTATS*. arXiv:2311.03260
- Platonov et al. (2023). "A critical look at the evaluation of GNNs under heterophily." *ICLR*. arXiv:2302.11640
