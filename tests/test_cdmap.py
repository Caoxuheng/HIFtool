from __future__ import annotations

import unittest
import warnings
from unittest import mock

import numpy as np

from Networks.CDMAP import CDMAP, CDMAPConfig


def synthetic_pair(scale=2, seed=7):
    rng = np.random.default_rng(seed)
    lr = rng.uniform(0.05, 0.95, size=(2, 3, 31)).astype(np.float32)
    srf = rng.uniform(0.01, 1.0, size=(31, 3)).astype(np.float32)
    srf /= srf.sum(axis=0, keepdims=True)
    hr_hsi = np.repeat(np.repeat(lr, scale, axis=0), scale, axis=1)
    hr_msi = np.matmul(hr_hsi, srf).astype(np.float32)
    return lr, hr_msi, srf


class CDMAPTests(unittest.TestCase):
    def test_forced_cpu_smoke(self):
        lr, hr, srf = synthetic_pair()
        model = CDMAP(CDMAPConfig(backend="cpu", sf=2), srf=srf)
        output = model(lr, hr)
        self.assertEqual(model.backend_name, "cpu")
        self.assertEqual(output.shape, (4, 6, 31))
        self.assertEqual(output.dtype, np.float32)
        self.assertTrue(np.isfinite(output).all())

    def test_auto_falls_back_to_cpu(self):
        lr, hr, srf = synthetic_pair()
        with mock.patch.object(CDMAP, "_probe_cuda", return_value=(False, "simulated no GPU")):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = CDMAP(CDMAPConfig(backend="auto", sf=2), srf=srf)
        self.assertEqual(model.backend_name, "cpu")
        self.assertTrue(any("fallback" in str(item.message).lower() or "CPU backend" in str(item.message)
                            for item in caught))
        self.assertTrue(np.isfinite(model(lr, hr)).all())

    def test_explicit_cuda_does_not_silently_fallback(self):
        _, _, srf = synthetic_pair()
        with mock.patch.object(CDMAP, "_probe_cuda", return_value=(False, "simulated no GPU")):
            with self.assertRaises(RuntimeError):
                CDMAP(CDMAPConfig(backend="cuda", sf=2), srf=srf)

    def test_cpu_cuda_parity_when_cuda_is_available(self):
        lr, hr, srf = synthetic_pair()
        available, _ = CDMAP._probe_cuda()
        if not available:
            self.skipTest("Numba CUDA is not available in this test environment")
        cpu = CDMAP(CDMAPConfig(backend="cpu", sf=2), srf=srf)(lr, hr)
        gpu = CDMAP(CDMAPConfig(backend="cuda", sf=2), srf=srf)(lr, hr)
        np.testing.assert_allclose(cpu, gpu, rtol=3e-3, atol=2e-3)


if __name__ == "__main__":
    unittest.main()
