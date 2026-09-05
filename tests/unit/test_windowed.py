import numpy as np

from partial_discharge_adaptive_fusion.windowed import (
    cwt_bag_batch, fit_window_standardizer, temporal_bag_batch, write_windowed_cache,
)
from partial_discharge_adaptive_fusion.windowing import WindowSpec


def test_windowed_temporal_cache_preserves_parent_and_window_shapes(tmp_path):
    raw = np.arange(20, dtype=np.float32).reshape(2, 10)
    factory = lambda: iter([raw])
    spec = WindowSpec(window_length=4, stride=4, aggregation="max")
    standardizer = fit_window_standardizer(factory, spec, representation="temporal")
    prepared = temporal_bag_batch(raw, spec, standardizer)
    assert prepared.shape == (2, 3, 1, 4)
    path = tmp_path / "temporal.npy"
    write_windowed_cache(
        factory, n_samples=2, spec=spec, representation="temporal", standardizer=standardizer,
        destination=str(path), dataset_id="dataset", dataset_version="v1", partition="train",
        dtype="float32",
    )
    cached = np.load(path, allow_pickle=False)
    assert cached.shape == prepared.shape
    metadata = path.with_suffix(".json").read_text()
    assert "representation-aware-mi-v1" in metadata


def test_windowed_cwt_cache_has_one_channel_and_fixed_scale_bins():
    raw = np.arange(32, dtype=np.float32).reshape(2, 16)
    factory = lambda: iter([raw])
    spec = WindowSpec(window_length=8, stride=8, aggregation="top_k_mean")
    standardizer = fit_window_standardizer(
        factory, spec, representation="cwt", scales=np.array([1.5, 4.0]), time_bins=6,
    )
    prepared = cwt_bag_batch(
        raw, spec, standardizer, scales=np.array([1.5, 4.0]), time_bins=6,
    )
    assert prepared.shape == (2, 2, 1, 2, 6)
    assert np.isfinite(prepared).all()
