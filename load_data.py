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
                 rna_path='data/tidy/train_rna.csv',
                 mismatch_path='data/tidy/sum_tab_1.csv',
                 test_proteomic_path='data/raw/test_pro.tsv',
                 test_clinical_path='data/raw/test_cli.tsv',
                 test_rna_path='data/raw/test_rna.tsv'):
        """Load the training data into pandas DataFrames.

        Keyword Arguments:
            clinical_path {str} -- The path to the clinical data.
                (default: {'data/tidy/train_cli.csv'})
            proteomic_path {str} -- The path to the proteomic data.
                Note that this will be normalized. (default:
                {'data/tidy/train_pro.csv'})
            mismatch_path {str} -- The path to the mismatch data.
                (default: {'data/tidy/sum_tab_1.csv'})
            test_proteomic_path {str}
        """
        self.clinical = pd.read_csv(clinical_path, index_col=0)
        self.proteomic = self.preprocess(
            pd.read_csv(proteomic_path, index_col=0)
        )
        self.rna = self.preprocess(
            pd.read_csv(rna_path, index_col=0)
        )
        self.mismatch = pd.read_csv(mismatch_path, index_col=0)
        self.test_proteomic = self.preprocess(
            pd.read_csv(test_proteomic_path, index_col=0, sep='\t').T
        )
        self.test_rna = self.preprocess(
            pd.read_csv(test_rna_path, index_col=0, sep='\t').T
        )
        self.test_clinical = pd.read_csv(test_clinical_path, index_col=0, sep='\t')

    def preprocess(self, df):
        return self.normalize(self.fix_data(df))

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
        return df.dropna(axis='columns', how='all').fillna(0.0)

if __name__ == "__main__":
    data = LoadData()
    print(data.rna)
