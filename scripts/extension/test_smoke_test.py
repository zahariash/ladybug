import unittest

from smoke_test import baseline_unreachable, counts_as_failure, engines_for_line, missing_from, validate_names

LOAD_ERROR = "exit code 1: Error: libfts.lbug_extension: undefined symbol: _ZN4lbug6common11ValueVector10setAllNullEv."
NOT_SERVED = 'exit code 1: RuntimeError: IO exception: HTTP Returns: 404, Failed to download extension: "fts" from x.'
UNREACHABLE = "exit code 1: RuntimeError: IO exception: Failed to download extension: fts at URL x (ERROR: Read)"


class CountsAsFailureTest(unittest.TestCase):
    def test_a_check_that_loads_never_counts(self):
        self.assertFalse(counts_as_failure(None, None))

    def test_a_failure_counts_without_a_baseline_or_when_the_baseline_loads(self):
        self.assertTrue(counts_as_failure(LOAD_ERROR, None))

    def test_a_failure_the_baseline_shares_does_not_count(self):
        self.assertFalse(counts_as_failure(LOAD_ERROR, LOAD_ERROR))

    def test_a_failure_does_not_count_when_the_baseline_does_not_serve_the_extension(self):
        self.assertFalse(counts_as_failure(LOAD_ERROR, NOT_SERVED))

    def test_an_unreachable_baseline_cannot_hide_a_failure(self):
        self.assertTrue(counts_as_failure(LOAD_ERROR, UNREACHABLE))
        self.assertTrue(counts_as_failure(LOAD_ERROR, "timed out after 300 s"))

    def test_baseline_unreachable(self):
        self.assertTrue(baseline_unreachable(UNREACHABLE))
        self.assertFalse(baseline_unreachable(NOT_SERVED))
        self.assertFalse(baseline_unreachable(LOAD_ERROR))


class MissingFromTest(unittest.TestCase):
    def test_an_extension_the_published_tree_serves_must_stay(self):
        self.assertEqual(missing_from(["fts"], ["fts", "json", "vector"]), ["json", "vector"])

    def test_nothing_is_missing_when_the_line_is_not_published_yet(self):
        self.assertEqual(missing_from(["fts"], []), [])
        self.assertEqual(missing_from([], []), [])


class InputsTest(unittest.TestCase):
    def test_non_release_lines_have_no_engines(self):
        self.assertEqual(engines_for_line("vdev", ["0.22.0"]), [])

    def test_extension_names_must_be_identifiers(self):
        self.assertEqual(validate_names(["fts", "unity_catalog"]), ["fts", "unity_catalog"])
        with self.assertRaises(SystemExit):
            validate_names(["fts' FROM 'http://evil/"])


if __name__ == "__main__":
    unittest.main()
