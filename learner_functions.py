"""
Contained in file are functions used for 
    - training scikit learn classifiers
    - making prediction with each algorithm
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

RAND_STATE = 0
TEST_SIZE = 0.3
NUMBER_OF_SPLITS = 10
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
    knn = KNeighborsClassifier()
    cv = StratifiedShuffleSplit( 
            n_splits = NUMBER_OF_SPLITS, 
            test_size = TEST_SIZE, 
            random_state = RAND_STATE )
    
    scores = cross_val_score(knn, data, labels, cv = cv, scoring = SCORING_METHOD)
    print(scores)
    
def train_sgd(data, labels):

    sgd = SGDClassifier(shuffle=True)
    sgd.fit(data, labels)

    cv = StratifiedShuffleSplit(
        n_splits=NUMBER_OF_SPLITS,
        test_size=TEST_SIZE,
        random_state=RAND_STATE)

    scores = cross_val_score(sgd, data, labels, cv=cv, scoring=SCORING_METHOD)
    print(scores)
