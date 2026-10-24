"""Offline checks for the paired bootstrap interval."""

import unittest

from bootstrap import paired_bootstrap


class BootstrapChecks(unittest.TestCase):
    def test_identical_systems_give_zero_difference_and_interval(self) -> None:
        result = paired_bootstrap([1, 2, 3, 4], [10] * 4, [1, 2, 3, 4], [10] * 4, resamples=200)
        self.assertEqual((result["difference"], result["ci_low"], result["ci_high"]), (0.0, 0.0, 0.0))

    def test_consistent_improvement_excludes_zero(self) -> None:
        worse = [5] * 40
        better = [2] * 40
        result = paired_bootstrap(worse, [10] * 40, better, [10] * 40, resamples=500, seed=1)
        self.assertAlmostEqual(result["difference"], -0.3)
        self.assertLess(result["ci_high"], 0)

    def test_is_deterministic_for_a_seed_and_validates_input(self) -> None:
        args = ([1, 5, 2, 8], [10] * 4, [2, 4, 3, 7], [10] * 4)
        self.assertEqual(paired_bootstrap(*args, resamples=300, seed=7),
                         paired_bootstrap(*args, resamples=300, seed=7))
        with self.assertRaisesRegex(ValueError, "same nonzero"):
            paired_bootstrap([1], [1], [1, 2], [1, 2])
        with self.assertRaisesRegex(ValueError, "level"):
            paired_bootstrap([1], [1], [1], [1], level=1.5)


if __name__ == "__main__":
    unittest.main()
