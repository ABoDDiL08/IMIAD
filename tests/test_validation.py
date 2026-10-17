"""Checks for disjoint, deterministic normal validation selection."""

import unittest

from imiad.validation import split_normal_train_validation


class ValidationSplitTest(unittest.TestCase):
    def test_uses_unselected_images_without_reducing_max_train(self) -> None:
        records = list(range(350))
        train, validation = split_normal_train_validation(records, 50, 0.2, 10, 42)
        self.assertEqual(train, list(range(50)))
        self.assertEqual(len(validation), 10)
        self.assertTrue(all(index >= 50 for index in validation))
        self.assertEqual(validation, split_normal_train_validation(records, 50, 0.2, 10, 42)[1])

    def test_minimum_validation_size_with_small_train_limit(self) -> None:
        train, validation = split_normal_train_validation(list(range(350)), 10, 0.2, 10, 42)
        self.assertEqual((len(train), len(validation)), (10, 10))

    def test_holds_out_training_records_when_no_unused_images_exist(self) -> None:
        train, validation = split_normal_train_validation(list(range(10)), None, 0.2, 10, 42)
        self.assertEqual((len(train), len(validation)), (8, 2))
        self.assertEqual(set(train) | set(validation), set(range(10)))
        self.assertFalse(set(train) & set(validation))

    def test_one_training_image_can_use_one_unused_validation_image(self) -> None:
        self.assertEqual(split_normal_train_validation([0, 1], 1, 0.2, 10, 42),
                         ([0], [1]))

    def test_single_available_image_cannot_calibrate_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            split_normal_train_validation([0], None, 0.2, 10, 42)


if __name__ == "__main__":
    unittest.main()
