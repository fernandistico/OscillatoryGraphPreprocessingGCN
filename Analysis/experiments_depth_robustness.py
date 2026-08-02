# experiments_depth_robustness

def depth_robustness_suite(datasets=['Cora', 'CiteSeer', 'PubMed']):
    """
    Table 2 & Figure 4: Main results
    
    For each dataset:
      For each depth L in [1, 2, 4, 6, 8, 10, 12, 16]:
        For each seed in range(8):
          - Train GCN with baseline features X
          - Train GCN with augmented features X_aug = [X | F_osc]
          - Record: accuracy, std, CKA w.r.t. frozen model
    
    Compute:
      - Delta = X_aug_acc - X_acc (percentage points)
      - Significance test (95% CI, z-score)
      - Across-seed std for stability (Fig 4c)
      - Linear CKA divergence (most distinct at peak gain)
    
    Returns: Table 2 (accuracies), Figure 4 data
    """

def compute_linear_cka(embeddings_baseline, embeddings_augmented):
    """
    Centered kernel alignment (Kornblith 2019, cited in paper).
    Paper use: validate that augmented and baseline diverge where
               gains are largest (CKA ≈ 0.11 at L=12 PubMed),
               converge at extreme depth (CKA ≈ 0.85 at L=16).
    """
