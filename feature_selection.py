"""
Explore scikitlearn's feature selection capabilities.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, SelectFdr, SelectFpr, RFECV, RFE
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
    return _supported_cols(df, feature_selector)

def variance(df, threshold=0.125):
    return _select_features(df, VarianceThreshold, threshold=threshold)

def univariate(features, labels, method=SelectKBest, metric=f_classif, **kwargs):
    labels = _squash_labels(labels)
    selected = method(metric, **kwargs).fit(features, labels)
    return _supported_cols(features, selected)

def elimination(features, labels, classifier, eliminator=RFE, **kwargs):
    labels = _squash_labels(labels)
    selected = eliminator(classifier, **kwargs).fit(features, labels)
    return _supported_cols(features, selected)

def _squash_labels(labels):
    return labels['gender'] + labels['msi']

def _supported_cols(features, selected):
    return features[features.columns[selected.get_support(indices=True)]]

if __name__ == "__main__":
    data = LoadData()
    print(elimination(data.proteomic, data.clinical, RandomForestClassifier()))
