"""
Load training data into dataframes.
"""

import pandas as pd

class LoadData(object):
    def __init__(self, 
                 clinical_path='data/tidy/train_cli.csv', 
                 proteomic_path='data/tidy/train_pro.csv', 
                 mismatch_path='data/tidy/sum_tab_1.csv'):
        self.clinical = pd.read_csv(clinical_path, index_col=0)
        self.proteomic = pd.read_csv(proteomic_path, index_col=0)
        self.mismatch = pd.read_csv(mismatch_path, index_col=0)

if __name__ == "__main__":    
    data_loader = LoadData()