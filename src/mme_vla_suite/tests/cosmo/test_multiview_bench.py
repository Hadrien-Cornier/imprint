"""CPU smoke test for the multiview latency bench contract."""
import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from scripts.bench_latency_multiview import run_benchmark


def test_bench_mv_keys():
    views = [1, 2]
    lengths = [64]
    rep = run_benchmark(views=views, buffer_lengths=lengths, iters=2, warmup=1,
                        use_fake=True)
    required = ["wan_encode_ms_perfire", "torch_to_jax_ms", "bottleneck_ms",
                "total_optionD_ms", "overhead_ms"]
    for V in views:
        for L in lengths:
            key = f"V{V}_L{L}"
            assert key in rep, f"missing row {key}"
            row = rep[key]
            for k in required:
                assert k in row, f"{key} missing {k}"
                assert isinstance(row[k], float)
            # V-scaling sanity: video KV token count grows with V.
            assert row["_meta"]["n_video_tokens"] == max(1, L // 4) * V * 64
    assert "_notes" in rep
