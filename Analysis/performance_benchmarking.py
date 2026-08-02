# performance_benchmarking

def benchmark_citation_graphs():
    """
    Figure 3a: CPU vs GPU timing on citation datasets
    Lenovo laptop, deterministic mode
    Report: speedup (GPU/CPU) per dataset
    """

def benchmark_synthetic_scaling():
    """
    Figure 3b: CPU and GPU scaling on synthetic graphs
    Wulver HPC cluster (NJIT)
    - Graph: fixed-degree-4 ring lattice
    - Sizes: N = 1000, 5000, 10000, ..., 100000 (5 repeats each)
    - Report: wall-clock time, log-log fit, CPU/GPU crossover
    Expected: GPU crossover near N ≈ 37,000 (measured 36,900)
    """

def compute_gpu_crossover():
    """
    Fit N* where GPU becomes faster than CPU.
    Paper reports: N* ≈ 36,900 (synthetic), first tested GPU-win at N=40k
    """
