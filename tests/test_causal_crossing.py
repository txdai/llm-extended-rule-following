"""Numerical invariants for source-message patching and experimental splits."""
import unittest

import torch

from counting.probes.suite.causal_crossing import cases, exact_integer, replace_source
from counting.probes.suite.causal_crossing_tasks import execute, generate_tasks, decision


class SourceMessageTests(unittest.TestCase):
    def test_replacement_matches_explicit_source_sum(self):
        torch.manual_seed(7)
        weights = torch.randn(1, 4, 9, 9, dtype=torch.float64).softmax(-1)
        donor_weights = torch.randn_like(weights).softmax(-1)
        values = torch.randn(1, 4, 9, 3, dtype=torch.float64)
        donor_values = torch.randn_like(values)
        source = 3
        output = (weights @ values).transpose(1, 2)
        donor = (donor_weights[..., source:source + 1]
                 * donor_values[:, :, source:source + 1, :]).transpose(1, 2)
        edited = replace_source(output, weights, values, source, donor)
        # Independently sum the untouched sources and one donor source.
        expected = torch.zeros_like(output)
        for k in range(9):
            contribution = (weights[..., k:k + 1] * values[:, :, k:k + 1, :]).transpose(1, 2)
            if k == source:
                contribution = donor
            expected += contribution
        torch.testing.assert_close(edited[:, source + 1:], expected[:, source + 1:], atol=3e-7, rtol=3e-7)
        self.assertTrue(torch.equal(edited[:, :source + 1], output[:, :source + 1]))

    def test_bfloat_sham_is_exact(self):
        torch.manual_seed(8)
        weights = torch.randn(1, 4, 9, 9).softmax(-1).bfloat16()
        values = torch.randn(1, 4, 9, 3).bfloat16()
        output = (weights @ values).transpose(1, 2)
        message = (weights[..., 3:4] * values[:, :, 3:4, :]).transpose(1, 2)
        self.assertTrue(torch.equal(output, replace_source(output, weights, values, 3, message)))

    def test_disjoint_count_splits(self):
        for model in ("gpt-oss-20b", "gemma-4-26b-a4b-it"):
            rows = cases(model)
            train = {r["n"] for r in rows if r["split"] == "calibration"}
            test = {r["n"] for r in rows if r["split"] == "heldout"}
            self.assertEqual((len(train), len(test)), (6, 8))
            self.assertFalse(train & test)
            self.assertEqual(len({r["id"] for r in rows}), 42)
            self.assertTrue(all(1 <= r["j"] < r["n"] for r in rows))

    def test_full_integer_scoring(self):
        for text, expected in [(" 18\n", 18), ("-1", -1), ("18 items", None),
                               ("1 8", None), ("", None), ("1", 1)]:
            self.assertEqual(exact_integer(text), expected)

    def test_noncommuting_histories_and_counterfactual_labels(self):
        from collections import Counter
        tasks = generate_tasks()
        self.assertEqual(len(tasks), 24)
        for a, b in zip(tasks[::2], tasks[1::2]):
            self.assertEqual(Counter(a["operations"]), Counter(b["operations"]))
            self.assertNotEqual(a["true_state"], b["true_state"])
        for task in tasks:
            self.assertEqual(execute(task["initial"], task["donor_operations"]), task["omitted_state"])
            for branch in ("different", "same"):
                spec = task["branches"][branch]
                true = decision(task["true_state"], spec["object"], spec["goal"], spec["plans"])
                omitted = decision(task["omitted_state"], spec["object"], spec["goal"], spec["plans"])
                self.assertEqual((true, omitted), (spec["true_answer"], spec["omitted_answer"]))
                self.assertEqual(true == omitted, branch == "same")


if __name__ == "__main__":
    unittest.main()
