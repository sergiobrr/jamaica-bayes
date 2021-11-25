from app import train
from unittest import TestCase
import numpy as np


class TestTrain(TestCase):

    def test_prepare_datasets(self):
        train_dataset, test_dataset = train.prepare_datasets()
        self.assertEqual(
            list(train_dataset.keys()),
            ['train_x', 'train_y']
        )

        self.assertEqual(
            list(test_dataset.keys()),
            ['test_x', 'test_y']
        )

        for train_key, test_key in zip(
            train_dataset.keys(), test_dataset.keys()
        ):
            self.assertEqual(
                8000, len(train_dataset[train_key])
            )
            self.assertEqual(
                2000, len(test_dataset[test_key])
            )

    def test_count_tweets(self):
        expected_result = {
            ('happi', 1): 1, ('trick', 0): 1,
            ('sad', 0): 1, ('tire', 0): 2
        }
        actual_result = {}
        tweets = [
            'i am happy', 'i am tricked', 
            'i am sad', 'i am tired', 'i am tired'
        ]
        ys = [1, 0, 0, 0, 0]
        train.count_tweets(
            actual_result, tweets, ys
        )
        self.assertDictEqual(
            expected_result,
            actual_result,
            f'{expected_result} è il conteggio delle parole per classe'
        )
