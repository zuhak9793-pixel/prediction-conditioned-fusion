"""Bounded synthetic checks; no external dataset or pretrained state required."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import expit
from pcf import fusion, b3, pcw, pipeline
from pcf.data import folds, load_triples, metrics, read_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = read_json(ROOT / 'configs/frozen.json')


class SyntheticChecks(unittest.TestCase):
    def test_frozen_settings(self):
        self.assertEqual(CONFIG['nimf'], dict(dimension=10, alpha=.4, TopK=10,
                                             **{'lambda': 30.}, etaInit=.01, iterations=95))
        self.assertEqual(CONFIG['B3'], dict(method='B3', **{'lambda': 10.},
                                           lambda1=10., tau=1., bias_lambda=.3))
        self.assertEqual(CONFIG['B3_sweeps'], 49)
        self.assertEqual(CONFIG['gate']['initial'], [0., -2.])
        self.assertEqual(CONFIG['smooth_delta_s'], .01)
        self.assertEqual(CONFIG['regularization'], 1e-4)

    def test_gate_formula_gradients_and_no_clipping(self):
        n, b = np.array([-4., .5, 2., 8.]), np.array([-2., 0., 1., 6.])
        y = np.array([.1, .7, 2.5, 7.])
        theta = np.array([.2, -1.3])
        p = fusion.forward('PCF', theta, n, b)[0]
        w = expit(.2 - np.logaddexp(0, -1.3) * np.log1p(np.maximum(b, 0)))
        np.testing.assert_array_equal(p, b + w * (n - b))
        self.assertLess(p[0], 0)
        self.assertTrue(np.all(p >= np.minimum(n, b)))
        self.assertTrue(np.all(p <= np.maximum(n, b)))
        grid = np.linspace(-2, 20, 100)
        weights = expit(theta[0] - np.logaddexp(0, theta[1]) * np.log1p(np.maximum(grid, 0)))
        self.assertTrue(np.all(np.diff(weights) <= 0))
        for kind, t in [('PCF', theta), ('constant_sigmoid', np.array([.13])),
                        ('B3_nonlinear', np.array([.13, -.2])),
                        ('unconstrained_sigmoid', theta)]:
            value, gradient = fusion.objective(kind, t, n, b, y, CONFIG)
            finite_difference = np.empty_like(t)
            for j in range(len(t)):
                step = np.zeros_like(t); step[j] = 1e-6
                finite_difference[j] = (fusion.objective(kind, t+step, n, b, y, CONFIG)[0]
                                        - fusion.objective(kind, t-step, n, b, y, CONFIG)[0]) / 2e-6
            np.testing.assert_allclose(gradient, finite_difference, atol=1e-8, rtol=1e-5)
            pred, _, penalty, _ = fusion.forward(kind, t, n, b)
            expected = np.mean(np.sqrt((pred-y)**2+.01**2)-.01) + .5e-4*penalty
            self.assertAlmostEqual(value, expected, places=14)

    def test_b3_directional_solve(self):
        opposite = np.array([[1., 2.], [3., -1.], [0., 4.]])
        adj = csr_matrix(np.array([[1., 1., 0.], [0., 1., 1.]]))
        residual = csr_matrix(np.array([[.2, .3, 0.], [0., -.2, .1]]))
        factors, grams, penalties = b3.block(opposite, adj, residual, np.array([10., 10.]), 10., 1.)
        for row, ix in enumerate(([0, 1], [1, 2])):
            gram = opposite[ix].T @ opposite[ix]
            penalty = 10*np.eye(2) + 10*np.linalg.inv(gram+np.eye(2))
            target = residual.toarray()[row] @ opposite
            np.testing.assert_allclose(grams[row], gram, atol=1e-14)
            np.testing.assert_allclose(penalties[row], penalty, atol=1e-14)
            np.testing.assert_allclose(factors[row], np.linalg.solve(gram+penalty, target), atol=1e-14)

    def test_metrics_retain_negative_and_extreme_errors(self):
        result = metrics(np.array([1., 2., 3.]), np.array([-1., 2., 13.]))
        self.assertEqual(result['mae'], 4.)
        self.assertEqual(result['rmse'], np.sqrt(104/3))
        with self.assertRaises(ValueError):
            metrics(np.ones(3), np.ones(1))

    def test_pure_roles_and_reproducible_folds(self):
        (ROOT / 'build').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'build') as tmp:
            path = Path(tmp) / 'evaluation.npz'
            # Missing y deliberately: a wrong role must be rejected before labels are accessed.
            np.savez(path, role='reserved-test')
            with self.assertRaises(ValueError):
                load_triples(path, 'training')
        self.assertEqual(np.bincount(folds(93691, 20261107)).tolist(), [31231, 31230, 31230])
        np.testing.assert_array_equal(folds(17, 20261108), folds(17, 20261108))

    def test_independent_preparation_recipe_on_synthetic_matrix(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('prepare_data', ROOT/'scripts/prepare_data.py')
        prepare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prepare)
        matrix = np.arange(400, dtype=float).reshape(20, 20)
        matrix[0, 0] = -1
        final, roles = prepare.partitions(matrix)
        self.assertEqual(len(final), int(.2*399))
        for seed, item in roles.items():
            self.assertEqual(len(item['validation']), int(.05*399))
            self.assertEqual(len(item['train01']), int(.01*399))
            np.testing.assert_array_equal(item['train01'], item['train05'][:len(item['train01'])])
            for role in item.values():
                self.assertFalse(np.intersect1d(final, role).size)
            self.assertFalse(np.intersect1d(item['validation'], item['train05']).size)

    def test_histogram_edges_and_fallback(self):
        h = pcw.histogram(np.array([0, 0, 0, 1]), np.array([0., 0., 35., 7.]), users=3)
        q = np.array([0., 35., 1., -1., 36., 7., 6., 0.])
        u = np.array([0, 0, 0, 0, 0, 1, 1, 2])
        np.testing.assert_allclose(pcw.confidence(h, u, q)[0], [2/3, 1/3, 0, 0, 0, 1, 0, 0])
        p, _ = pcw.fuse(h, np.array([0, 0, 2]), np.array([0., -1., 2.]), np.array([35., 36., 4.]))
        np.testing.assert_allclose(p, [35/3, 17.5, 3.])

    def test_oof_omitted_labels_never_enter_experts(self):
        u, s = np.repeat(np.arange(3), 4), np.tile(np.arange(4), 3)
        y = np.arange(12, dtype=float)+1
        calls = []
        def fake_fit(train, coordinates, shape, config, seed):
            tu, ts, ty = train
            hu, hs = coordinates
            self.assertFalse(np.intersect1d(tu*4+ts, hu*4+hs).size)
            np.testing.assert_array_equal(ty, y[tu*4+ts])
            calls.append(len(ty))
            return np.ones(len(hu)), np.ones(len(hu)), None, None
        with tempfile.TemporaryDirectory(dir=ROOT / 'build') as tmp:
            with patch.object(pipeline, 'expert_predictions', fake_fit):
                pipeline.make_oof((u, s, y), (3, 4), CONFIG, 20261107, Path(tmp)/'oof.npz')
        self.assertEqual(calls, [8, 8, 8])

    def test_native_objective_gradient_and_owned_factors(self):
        from NIMF import Model
        rng = np.random.RandomState(73)
        matrix = rng.uniform(.1, 2, (5, 7)); matrix[0, :2] = 0
        p, q = rng.rand(5, 10), rng.rand(7, 10)
        native = Model(matrix, p, q, 30., .4, .01)
        gp, gq = native.gradient()
        for which, row, col in [(0, 0, 1), (1, 2, 3)]:
            values = []
            for sign in (-1, 1):
                pp, qq = p.copy(), q.copy()
                (pp if which == 0 else qq)[row, col] += sign*1e-6
                values.append(Model(matrix, pp, qq, 30., .4, .01).objective())
            self.assertAlmostEqual((values[1]-values[0])/2e-6, (gp if which == 0 else gq)[row, col], delta=1e-6)
        old = native.factors()
        p[:] = 999
        np.testing.assert_array_equal(native.factors()[0], old[0])
        native.step()
        self.assertTrue(np.isfinite(native.predict(np.array([0, 1]), np.array([2, 3]))).all())

    def test_actual_oof_fit_full_infer_and_cli_score(self):
        from pcf.cli import main
        import sys
        u, s = np.repeat(np.arange(7), 9), np.tile(np.arange(9), 7)
        y = 1 + .1*u + .03*s + .04*np.sin(u+s)
        shape = (8, 10)  # retain unsupported user/service in full inference
        train = (u, s, y)
        with tempfile.TemporaryDirectory(dir=ROOT / 'build') as tmp:
            folder = Path(tmp)
            np.savez(folder/'train.npz', role='training', shape=shape, seed=20261107, u=u, s=s, y=y)
            commands = [
                ['oof', '--train', str(folder/'train.npz'), '--seed', '20261107', '--out', str(folder/'oof.npz')],
                ['fit-gate', '--oof', str(folder/'oof.npz'), '--out', str(folder/'gate.json'), '--development-ablation'],
                ['train', '--train', str(folder/'train.npz'), '--seed', '20261107', '--out', str(folder/'models')]]
            for command in commands:
                with patch.object(sys, 'argv', ['pcf', *command]):
                    main()
            rows, cols = np.array([7, 0, 7]), np.array([0, 9, 9])
            np.savez(folder/'coords.npz', u=rows, s=cols)
            with patch.object(sys, 'argv', ['pcf', 'predict', '--train', str(folder/'train.npz'),
                '--models', str(folder/'models'), '--gate', str(folder/'gate.json'),
                '--coords', str(folder/'coords.npz'), '--out', str(folder/'predictions.npz')]):
                main()
            np.savez(folder/'validation.npz', role='development-validation', shape=shape,
                     u=rows, s=cols, y=np.array([1., 2., 3.]))
            with patch.object(sys, 'argv', ['pcf', 'score', '--role', 'development-validation',
                '--targets', str(folder/'validation.npz'), '--predictions', str(folder/'predictions.npz'),
                '--out', str(folder/'metrics.json')]):
                main()
            score = read_json(folder/'metrics.json')
            self.assertEqual(score['metrics']['PCF']['n'], 3)
            with np.load(folder/'oof.npz') as z:
                self.assertTrue(np.isfinite(z['NIMF']).all())


if __name__ == '__main__':
    unittest.main()
