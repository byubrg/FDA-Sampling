"""
Explore scikitlearn's feature selection capabilities.
"""
from sklearn.feature_selection import VarianceThreshold
from load_data import LoadData

def normalize(df):
    """Normalize each column into roughly [-1.0, 1.0] centered around 0.0.

    Arguments:
        df {pandas.DataFrame} -- The data to normalize. Each column
            must be quantitative.

    Returns:
        pandas.DataFrame -- The normalized data.
    """

    return (df - df.mean()) / (df.max() - df.min())

def fix_data(df):
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

def select_features(df, selector=VarianceThreshold, **kwargs):
    """Select features to be used for classification based on some
    scikit-learn feature selector.

    Arguments:
        df {pandas.DataFrame} -- The DataFrame from which we select
            columns.

    Keyword Arguments:
        selector {scikit-learn feature selector} -- The feature
            selection class to use for feature selection
            (default: {VarianceThreshold})

    Returns:
        list -- The names of the selected features (columns in the
            input DataFrame).
    """
    df = normalize(df)
    feature_selector = selector(**kwargs).fit(df)
    return df.columns[feature_selector.get_support(indices=True)]

if __name__ == "__main__":
    data = LoadData()
    proteomic = fix_data(data.proteomic)
    print(select_features(proteomic, VarianceThreshold, threshold=0.125))
