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
                 train_proteomic_path='data/raw/train_pro.tsv',
                 test_proteomic_path='data/raw/test_pro.tsv',
                 train_clinical_path='data/raw/train_cli.tsv',
                 test_clinical_path='data/raw/test_cli.tsv',
                 train_rna_path='data/raw/train_rna.tsv',
                 test_rna_path='data/raw/test_rna.tsv',
                 mislabel_path='data/tidy/sum_tab_2.csv'):

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
        #There are test proteomic, rna, and clinical, but not train clinical or ptoteomic
        self.test_proteomic = self.preprocess(
            pd.read_csv(test_proteomic_path, index_col=0, sep='\t').T
        )
        self.test_rna = self.preprocess(
            pd.read_csv(test_rna_path, index_col=0, sep='\t').T
        )
        self.train_proteomic = pd.read_csv(train_proteomic_path, index_col=0, sep='\t') #
        self.test_proteomic = pd.read_csv(test_proteomic_path, index_col=0, sep='\t') #
        self.train_clinical = pd.read_csv(train_clinical_path, index_col=0, sep='\t') #
        self.test_clinical = pd.read_csv(test_clinical_path, index_col=0, sep='\t')
        self.train_rna = pd.read_csv(train_rna_path, index_col=0, sep='\t').T
        self.test_rna = pd.read_csv(test_rna_path, index_col=0, sep='\t').T
        #why are there two test.rna?
        #If we make new ones, do we use the preprocessing?
        #Can I make new parts for proteomic and clinical data?
        self.train_pro_cli = self.train_clinical.merge(self.proteomic, how='outer', left_index=True, right_index=True) #
        self.test_pro_cli = self.test_clinical.merge(self.proteomic, how='outer', left_index=True, right_index=True) #
        self.train_rna_cli = self.train_clinical.merge(self.rna, how='outer', left_index=True, right_index=True) #
        self.test_rna_cli = self.test_clinical.merge(self.rna, how='outer', left_index=True, right_index=True) #
        self.train_pro_rna = self.train_rna.merge(self.proteomic, how='outer', left_index=True, right_index=True)
        self.test_pro_rna = self.test_rna.merge(self.test_proteomic, how='outer', left_index=True, right_index=True)
        #Can I make a train_pro_rna?
        self.train_all = self.train_pro_rna.merge(self.clinical, how='outer', left_index=True, right_index=True)
        self.train_all = self.train_all.replace(['Female', 'Male','MSI-Low/MSS', 'MSI-High'], [0, 1, 0, 1])

        self.test_all = self.test_pro_rna.merge(self.test_clinical, how='outer', left_index=True, right_index=True)
        self.test_all = self.test_all.replace(['Female', 'Male', 'MSI-Low/MSS', 'MSI-High'], [0, 1, 0, 1])

        self.mislabel = pd.read_csv(mislabel_path, index_col=0)

        # create training labels for if a sample has been mislabeled
        self.mislabel_labels = []
        for i in range(0, len(self.mislabel.index)):
            if self.mislabel.iloc[i, 0] == self.mislabel.iloc[i, 1] and self.mislabel.iloc[i, 1] == self.mislabel.iloc[i, 2]:
                self.mislabel_labels.append(0)
            else:
                self.mislabel_labels.append(1)


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
