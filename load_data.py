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
                (default: {'data/tidy/train_pro.csv'})
            mismatch_path {str} -- The path to the mismatch data.
                (default: {'data/tidy/sum_tab_1.csv'})
        """
        self.clinical = pd.read_csv(clinical_path, index_col=0)
        self.proteomic = pd.read_csv(proteomic_path, index_col=0)
        self.mismatch = pd.read_csv(mismatch_path, index_col=0)

if __name__ == "__main__":
    pass
