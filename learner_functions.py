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
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

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

def train_lr(data, labels):
    train_classifier(data, labels, LogisticRegression)

def train_knn(data,labels):
    train_classifier(data,labels, KNeighborsClassifier)
    
def train_sgd(data, labels):
    train_classifier(data,labels, SGDClassifier)
    
def train_nc(data,labels):
    train_classifier(data,labels, NearestCentroid)

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

def train_svm(data, labels):
   train_classifier(data,labels, svm.SVC)

