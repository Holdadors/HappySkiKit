import unittest
from happy_ski import happyski
import pandas as pd
from unittest.mock import patch, MagicMock

class TestHappyski(unittest.TestCase):
    """
    A class that tests all the set up functions in the happyski class
    """
    @patch('pandas.read_csv') # Replace I/O operations so unit tests wont rely on
    # file system
    def setUp(self, mock_read_csv):
        # Mock read_csv to prevent actual file I/O
        mock_read_csv.return_value = pd.DataFrame({  # Create a pd_dataframe with all necessary data columns
            'Date': ['2021-01-01', '2021-01-02'],
            'Resort Name': ['Palisades Tahoe Resort', 'Mammoth Mountain Resort'],
            'Temp High (°F)': ['46.4', '28.7'],
            'Temp Low (°F)': ['32.3', '17'],
            'Snowfall (inches)': ['Alpine', 'Boulder'],
            'Wind (mph)': ['13.1', '10.9'],
            'Skies': ['Rainy', 'Snowy'],
            'Open Trails  (%)': ['89', '100'],
            'Snow Conditions': ['Machine Groomed', 'Powder'],
            'Score': [2, 5],
        })
        self.happyski = happyski('snow.csv', 'snow prediction.csv')

    def test_init(self):
        # Test initialization and file reading
        self.assertIsInstance(self.happyski.dataset, pd.DataFrame)
        self.assertIsNone(self.happyski.model)

    @patch('builtins.print')  # to suppress print statements during tests
    def test_clean_data(self, mock_print):
        # Testing clean data function
        df = pd.DataFrame({
            'Date': ['2021-01-01', '2021-01-02'],
            'Resort Name': ['Palisades Tahoe Resort', 'Mammoth Mountain Resort'],
            'Temp High (°F)': ['46.4', '28.7'],
            'Temp Low (°F)': ['32.3', '17'],
            'Snowfall (inches)': ['Alpine', 'Boulder'],
            'Wind (mph)': ['13.1', '10.9'],
            'Skies': ['Rainy', 'Snowy'],
            'Open Trails  (%)': ['89', '100'],
            'Snow Conditions': ['Machine Groomed', 'Powder'],
            'Score': [2, 5],
        })
        cleaned_df = self.happyski.clean_data(df)
        self.assertTrue('Date' not in cleaned_df.columns)
        self.assertTrue('Resort Name' not in cleaned_df.columns)
        self.assertTrue('Skies_Snowy' in cleaned_df.columns)

    @patch('sklearn.model_selection.train_test_split')
    def test_split_data(self, mock_train_test_split):
        # Testing split data
        mock_train_test_split.return_value = (None, None, None, None)
        result = self.happyski.split_data()
        # Testing # of data splits
        self.assertEqual(len(result), 4)
        # Testing if "Score" column is dropped for X
        self.assertTrue('Score' not in result[0].columns)

if __name__ == '__main__':
    unittest.main()