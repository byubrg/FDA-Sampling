"""
Explore scikitlearn's feature selection capabilities.
"""
from sklearn.feature_selection import VarianceThreshold
from load_data import LoadData

def normalize(df):
    return (df - df.mean()) / (df.max() - df.min())

def fix_data(df):
    bad_columns = ["TMEM35A"]
    return df.fillna(0.0).drop(bad_columns, axis="columns")

def select_features(df, selector=VarianceThreshold, **kwargs):
    df = normalize(df)
    feature_selector = selector(**kwargs).fit(df)
    return df.columns[feature_selector.get_support(indices=True)]

if __name__ == "__main__":
    data = LoadData()
    proteomic = fix_data(data.proteomic)
    print(select_features(proteomic, VarianceThreshold, threshold=0.125))
