from app import utils
from unittest import TestCase


class TestUtils(TestCase):

    def setUp(self) -> None:
        self.freqs = {
            ('sad', 0): 4,
            ('happy', 1): 12,
            ('oppressed', 0): 7
        }
        self.raw_tweet = "RT @Twitter @chapagain Hello There! Have a great day. :) #good #morning http://chapagain.com.np"
        return super().setUp()

    def test_lookup(self):
        expedted_value = 12
        returned_value = utils.lookup(
            self.freqs, 'happy', 1
        )
        self.assertEqual(
            expedted_value, returned_value,
            f'{expedted_value} is the positive freq for happy'
        )

    def test_process_tweet(self):
        expected_wvec = [
            'hello', 'great', 'day',
            ':)', 'good', 'morn'
        ]
        actual_wvec = utils.process_tweet(
            self.raw_tweet
        )
        self.assertEqual(
            expected_wvec,
            actual_wvec,
            f'{self.raw_tweet} dovrebbe essere stemmato, stop_words e punteggiatura'
        )
