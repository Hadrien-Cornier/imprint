# Training Container

Build from the repository root:

```bash
docker build -f envs/training/Dockerfile -t imprint-training:cuda12 .
```

Run with datasets, Pi0.5, and output directories mounted explicitly. The image
does not contain credentials, checkpoints, simulator code, or datasets.
