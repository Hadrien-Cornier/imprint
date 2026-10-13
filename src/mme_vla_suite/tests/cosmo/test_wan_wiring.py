def test_selector_routes_wan(monkeypatch):
    from mme_vla_suite.policies import policy as P

    class _Cosmo:
        def __init__(self, **k):
            self.kind = "cosmo"

    class _Wan:
        def __init__(self, **k):
            self.kind = "wan"
            self.temporal_factor = 4
            self.latent_dim = 48

    import mme_vla_suite.models.representation.cosmo_frontend as CF
    monkeypatch.setattr(CF, "CosmoFrontend", _Cosmo)
    import mme_vla_suite.models.representation.wan_frontend as WF
    monkeypatch.setattr(WF, "WanFrontend", _Wan)
    assert P._load_cosmo_frontend("CV8x8x8").kind == "cosmo"
    assert P._load_cosmo_frontend("Wan2.2-TI2V-5B").kind == "wan"
