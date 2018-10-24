"""
Explore scikitlearn's feature selection capabilities.
"""
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from load_data import LoadData
def _select_features(df, selector=VarianceThreshold, **kwargs):
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
    feature_selector = selector(**kwargs).fit(df)
    return df[df.columns[feature_selector.get_support(indices=True)]]

def select(df, threshold=0.125):
    return _select_features(df, VarianceThreshold, threshold=threshold)

if __name__ == "__main__":
    data = LoadData()
    #print(select(data.proteomic, threshold=0.125))
    selected = SelectKBest(f_classif, k=10).fit(data.proteomic, data.clinical["gender"]+data.clinical["msi"])
    df = data.proteomic
    print(df[df.columns[selected.get_support(indices=True)]])