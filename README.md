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

