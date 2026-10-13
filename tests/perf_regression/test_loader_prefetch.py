import numpy as np

from openpi.training.data_loader import TorchDataLoader


class _Dataset:
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {"value": np.asarray(index, dtype=np.int32)}


def test_torch_loader_accepts_configured_prefetch_factor():
    loader = TorchDataLoader(
        _Dataset(),
        local_batch_size=2,
        num_workers=1,
        prefetch_factor=4,
        framework="pytorch",
    )

    assert loader.torch_loader.prefetch_factor == 4
