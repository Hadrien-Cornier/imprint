"""Fail-loud gate: finalized full-pass == finalized prefix == streaming for Wan2.2 VAE."""
import sys, numpy as np
from mme_vla_suite.models.representation.wan_frontend import WanFrontend


def main():
    fe = WanFrontend(device="cuda")
    rng = np.random.default_rng(0)
    full = rng.integers(0, 256, (40, 224, 224, 3)).astype(np.uint8)
    fin = fe.finalized_encode(full)
    print("full finalized chunks:", fin.shape)

    # Causality: first 2 finalized chunks from full pass == finalized_encode(prefix)
    prefix = fe.finalized_encode(full[:24])  # 24//4 - 1 = 5 finalized chunks
    n = min(2, prefix.shape[0])
    d_prefix = np.abs(fin[:n] - prefix[:n]).max()
    ok_prefix = bool(d_prefix <= 1e-2)

    # Streaming equivalence
    stream = fe.streaming_encode(full, fe.temporal_factor)
    d_stream = np.abs(fin - stream).max()
    ok_stream = bool(d_stream <= 2e-2)

    print("finalized_causal=%s (max_abs_diff=%.6e)" % (ok_prefix, d_prefix))
    print("streaming_equiv=%s (max_abs_diff=%.6e)" % (ok_stream, d_stream))
    print("latent_dim=%d temporal_factor=%d spatial_tokens=%d"
          % (fe.latent_dim, fe.temporal_factor, fin.shape[1]))
    verdict = ok_prefix and ok_stream
    print("VERDICT:", "PASS" if verdict else "FAIL")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
