---
library_name: pytorch
tags:
  - robotics
  - rmbench
  - molmoact2
  - vision-language-action
---

# IMPRINT RMBench MolmoAct2

This repository contains the model-only 100k Stage-2 checkpoint used for the
IMPRINT MolmoAct2 RMBench experiment. Optimizer and trainer state are excluded.

The checkpoint imports the full-attention IMPRINT Stage-1 memory encoder,
finetunes the memory encoder and MolmoAct2 action policy, and uses 256 memory
tokens. Evaluation uses online WAN and DINOv3 history features plus the stored
Pi0.5 language embeddings distributed separately in
`Amazon-FAR/imprint_datasets` under
`simulation/rmbench/molmoact2_eval_assets/`.

Use the `molmoact2` branch of the IMPRINT repository for training and
closed-loop evaluation.
