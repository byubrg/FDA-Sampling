"""
Contained in file are functions used for
    - training scikit learn classifiers
    - making prediction with each algorithm
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix

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
    return model.fit(data, labels)


def train_rf(data, labels):
    return train_classifier(data, labels, RandomForestClassifier, n_estimators=5)

def train_lr(data, labels):
    return train_classifier(data, labels, LogisticRegression)

def train_knn(data,labels):
    return train_classifier(data,labels, KNeighborsClassifier)

def train_sgd(data, labels):
    return train_classifier(data,labels, SGDClassifier)

def train_nc(data,labels):
    return train_classifier(data,labels, NearestCentroid)

def train_mlp(data, labels):
    return train_classifier(data, labels, MLPClassifier, max_iter=300, solver='sgd')


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
   return train_classifier(data,labels, svm.SVC)

def make_test_prediction(model, data, labels, print_details=True):
    pred = model.predict(data)
    if(print_details):
        print('score', accuracy_score(pred, labels))
        print('pred', pred)
        print('actual', labels)
        print(confusion_matrix(labels,pred))

    return pred

"""
given the protein data, two model trained for gender and msi classification and the final sample names
writes to subchallenge_1.csv, the submission file
writes rows with sample id and if it is mismatched, denoted as a 0 or 1
mismatches are considered any instance where the predicted and given labels do not match
"""
def generate_and_write_results(pro_data, model_gender, model_msi, gender_labels, msi_labels, sample_names):
    gender_predictions = make_test_prediction(model_gender,pro_data,gender_labels)
    msi_predictions = make_test_prediction(model_msi,pro_data,msi_labels)
    outfile = open('subchallenge_1.csv','w')
    outfile.write('sample,mismatch\n')

    for i in range(0,len(msi_labels)):
        outfile.write(sample_names[i] + ',')
        if gender_labels[i] == gender_predictions[i] and msi_labels[i] == msi_predictions[i]:
            outfile.write('0\n')
        else:
            outfile.write('1\n')

    outfile.close()
