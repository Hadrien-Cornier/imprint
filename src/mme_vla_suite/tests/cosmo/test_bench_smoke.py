def test_bench_report_keys():
    from mme_vla_suite.scripts import bench_latency as B
    rep = B.run_benchmark(buffer_lengths=[8, 16], iters=3, warmup=1, use_fake_cosmos=True)
    row = rep[16]
    for k in ["cosmos_encode_ms", "siglip_recent_ms", "torch_to_jax_ms",
              "bottleneck_ms", "modulation_ms", "pi05_forward_ms",
              "total_optionD_ms", "overhead_ms"]:
        assert k in row and row[k] >= 0.0
