import os
import sys
import unittest
from unittest.mock import MagicMock
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

this_dir = os.path.dirname(__file__)
utils_dir = os.path.join(this_dir, '..')
sys.path.append(utils_dir)

from pre_processing.pre_process import PRE_PROCESS


class TestStatusPredictor(unittest.TestCase):
    def setUp(self):
        self.model = MagicMock(spec=RandomForestClassifier)
        self.df = pd.DataFrame(columns=['externalStatus_A', 'externalStatus_B', 'externalStatus_C', 'internalStatus'])
        self.file = [{'externalStatus': 'A', 'internalStatus': 'X'},
                     {'externalStatus': 'B', 'internalStatus': 'Y'},
                     {'externalStatus': 'C', 'internalStatus': 'Z'}]
        self.predictor = PRE_PROCESS(self.model, self.df, self.file)
        self.model.predict.return_value = ['X']

    def test_predict_internal_status(self):
        print("Testing with status 'A' which exists in the dataset.")
        result = self.predictor.predict_internal_status('A')
        print(f"Result for 'A': {result}")
        self.model.predict.assert_called_once()
        expected_output = ('X', 1.0, 1.0, 1.0)
        self.assertEqual(result, expected_output)
        self.model.predict.reset_mock()
        print("Testing with status 'C' which does not exist in the dataset.")
        result = self.predictor.predict_internal_status('D')
        print(f"Result for 'D': {result}")
        self.model.predict.assert_called_once()
        self.assertTrue(isinstance(result[1], float))
        self.assertTrue(isinstance(result[2], float))
        self.assertTrue(isinstance(result[3], float))


if __name__ == '__main__':
    unittest.main()
