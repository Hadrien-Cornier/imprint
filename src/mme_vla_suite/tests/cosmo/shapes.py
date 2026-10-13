"""Realistic shapes for Option-D synthetic tensors, single source of truth for tests + benches."""
# Cosmos CV8x8x8 on 224x224 frames -> ~ (t/8) chunks, each 28x28=784 spatial latent tokens, latent_dim=16 (CV continuous)
COSMOS_LATENT_DIM = 16          # Cosmos CV continuous latent channel dim
COSMOS_SPATIAL_TOKENS = 784     # 28*28 grid tokens per chunk at 224px, 8x spatial
COSMOS_TEMPORAL_FACTOR = 8      # frames per consolidated chunk (downsampled frames)
SIGLIP_TOKENS_PER_FRAME = 64    # existing SigLIP pooled 8x8
SIGLIP_DIM = 2048
DINOV3_TOKENS_PER_FRAME = 64    # pooled 8x8 DINOv3 patch grid
DINOV3_DIM = 768
STATE_DIM = 16                  # robocasa eef-first proprio state contract (Z_p base)
PROPRIO_IN_DIM = 32             # proprio + Δp (state 16 + delta 16) fed to MLP tokenizer
ACTION_IN_DIM = 24              # executed action 12 + Δa 12 fed to MLP tokenizer
LANG_EMB_DIM = 2048             # PaliGemma gemma_2b text embedding width (language-conditions queries)
BOTTLENECK_TOKENS = 256         # fixed memory budget (doc default); ablate 128/256/512
MEMORY_TOKEN_DIM = 1024         # modulation seam width
MAX_HISTORY_STEPS = 2048        # cap
DOWNSAMPLE = 2                  # default history downsample

NUM_VIEWS = 3                    # unified_lerobot has 3 camera views
WAN_LATENT_DIM = 48
WAN_TEMPORAL_FACTOR = 4
WAN_SPATIAL_TOKENS_128 = 64      # 8x8 grid at 128px input (16x spatial downsample)
