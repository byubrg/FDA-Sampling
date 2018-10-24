"""
Load training data into dataframes.
"""

import pandas as pd

class LoadData(object):
    """A class to load and hold the training data.
    """
    def __init__(self,
                 clinical_path='data/tidy/train_cli.csv',
                 proteomic_path='data/tidy/train_pro.csv',
                 mismatch_path='data/tidy/sum_tab_1.csv'):
        """Load the training data into pandas DataFrames.

        Keyword Arguments:
            clinical_path {str} -- The path to the clinical data.
                (default: {'data/tidy/train_cli.csv'})
            proteomic_path {str} -- The path to the proteomic data.
                Note that this will be normalized. (default:
                {'data/tidy/train_pro.csv'})
            mismatch_path {str} -- The path to the mismatch data.
                (default: {'data/tidy/sum_tab_1.csv'})
        """
        self.clinical = pd.read_csv(clinical_path, index_col=0)
        self.proteomic = pd.read_csv(proteomic_path, index_col=0)
        self.proteomic = self.fix_data(self.proteomic)
        self.proteomic = self.normalize(self.proteomic)
        self.mismatch = pd.read_csv(mismatch_path, index_col=0)

    def normalize(self, df):
        """Normalize each column into roughly [-1.0, 1.0] centered around 0.0.

        Arguments:
            df {pandas.DataFrame} -- The data to normalize. Each column
                must be quantitative.

        Returns:
            pandas.DataFrame -- The normalized data.
        """
        return (df - df.mean()) / (df.max() - df.min())

    def fix_data(self, df):
        """Preprocess dataframe to fill NaNs with 0s and remove bad
        columns.

        Arguments:
            df {pandas.DataFrame} -- DataFrame to be processed.

        Returns:
            pandas.DataFrame -- Processed dataframe. Note that some columns
                may be removed.
        """
        bad_columns = ["TMEM35A"]
        return df.fillna(0.0).drop(bad_columns, axis="columns")

if __name__ == "__main__":
    pass
