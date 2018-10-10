"""
Contained in file are functions used for
    - training scikit learn classifiers
    - making prediction with each algorithm
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import BaggingClassifier

RAND_STATE = 0
TEST_SIZE = 0.1
NUMBER_OF_SPLITS = 100
SCORING_METHOD = 'accuracy'

def train_classifier(data, labels, classifier, **kwargs):
    model = classifier(**kwargs)
    cv = StratifiedShuffleSplit(
        n_splits=NUMBER_OF_SPLITS,
        test_size=TEST_SIZE,
        random_state=RAND_STATE
    )
    scores = cross_val_score(model, data, labels, cv=cv, scoring=SCORING_METHOD)
    print(scores)

def train_rf(data, labels):
    train_classifier(data, labels, RandomForestClassifier, n_estimators=5)

def train_knn(data,labels):
    knn = KNeighborsClassifier(metric='manhattan')
    cv = StratifiedShuffleSplit(
            n_splits = NUMBER_OF_SPLITS,
            test_size = TEST_SIZE,
            random_state = RAND_STATE )

    scores = cross_val_score(knn, data, labels, cv = cv, scoring = SCORING_METHOD)
    print(scores)

def train_sgd(data, labels):

    sgd = SGDClassifier()
    sgd.fit(data, labels)

    cv = StratifiedShuffleSplit(
        n_splits=NUMBER_OF_SPLITS,
        test_size=TEST_SIZE,
        random_state=RAND_STATE)

    scores = cross_val_score(sgd, data, labels, cv=cv, scoring=SCORING_METHOD)
    print(scores)

def train_nc(data,labels):
    nc = NearestCentroid(metric='manhattan')
    cv = StratifiedShuffleSplit(
            n_splits = NUMBER_OF_SPLITS,
            test_size = TEST_SIZE,
            random_state = RAND_STATE )

    scores = cross_val_score(nc, data, labels, cv = cv, scoring = SCORING_METHOD)
    print(scores)

def train_bagging_knn(data,labels):
    bagging = BaggingClassifier(KNeighborsClassifier(metric='manhattan',algorithm='brute'),
                                n_estimators=30,
                                max_samples=0.25,
                                max_features=0.25,
                                warm_start=True)
    cv = StratifiedShuffleSplit(
        n_splits = NUMBER_OF_SPLITS,
        test_size = TEST_SIZE)

    scores = cross_val_score(bagging, data, labels, cv = cv, scoring = SCORING_METHOD)
    print(scores)
