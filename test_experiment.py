from unittest import TestCase

import numpy as np

from experiment import METHODS, exact_p, log_e, simulate


class ExperimentTests(TestCase):
    def test_known_exact_test(self):
        self.assertEqual(exact_p(0, 0), 1)
        self.assertAlmostEqual(exact_p(6, 0), 2 / 64)
        self.assertEqual(exact_p(6, 0), exact_p(0, 6))

    def test_mixture_satisfies_one_step_martingale_identity(self):
        for wins in range(8):
            for losses in range(8):
                current = np.exp(log_e(wins, losses))
                expected = (np.exp(log_e(wins + 1, losses)) + np.exp(log_e(wins, losses + 1))) / 2
                self.assertAlmostEqual(current, expected)

    def test_single_look_repeated_test_equals_fixed_test(self):
        result = simulate(np.random.default_rng(42), 1000, [100], 0.05, 0.20, 0.27)
        self.assertEqual(result[METHODS[0]], result[METHODS[1]])
