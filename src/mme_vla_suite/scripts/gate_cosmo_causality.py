"""Fail-loud gate: finalized full-pass == finalized prefix == streaming for the chosen Cosmos variant."""
import sys, numpy as np
from mme_vla_suite.models.representation.cosmo_frontend import CosmoFrontend

def main(variant="CV8x8x8"):
    fe = CosmoFrontend(variant=variant, device="cuda")
    rng = np.random.default_rng(0)
    full = rng.integers(0, 256, (40, 224, 224, 3)).astype(np.uint8)
    fin = fe.finalized_encode(full)
    ok_prefix = np.allclose(fin[:2], fe.finalized_encode(full[:24]), rtol=1e-2, atol=1e-2)
    ok_stream = np.allclose(fin, fe.streaming_encode(full, fe.temporal_factor), rtol=2e-2, atol=2e-2)
    print(f"variant={variant} finalized_causal={ok_prefix} streaming_equiv={ok_stream}")
    sys.exit(0 if (ok_prefix and ok_stream) else 1)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "CV8x8x8")
