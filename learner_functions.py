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
from sklearn.calibration import CalibratedClassifierCV
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
import hard_vote as hv

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
    print( sum(scores) / len(scores))
    score = sum(scores) / len(scores)
    return model.fit(data, labels), score


def train_rf(data, labels, **kwargs):
    return train_classifier(data, labels, RandomForestClassifier, **kwargs)


def train_lr(data, labels):
    return train_classifier(data, labels, LogisticRegression)


def train_knn(data,labels, **kwargs):
    return train_classifier(data,labels, KNeighborsClassifier, **kwargs)


def train_sgd(data, labels):
    return train_classifier(data,labels, SGDClassifier)


def train_sgd_mod(data, labels):
    return train_classifier(data,labels, SGDClassifier, loss='modified_huber')


def train_nc(data,labels,**kwargs):
    return train_classifier(data,labels, NearestCentroid,**kwargs)


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

    scores = cross_val_score(bagging, data, labels, cv=cv, scoring=SCORING_METHOD)
    print(scores)


def train_svm(data, labels,**kwargs):
   return train_classifier(data,labels, svm.SVC,**kwargs)


def make_test_prediction(model, data, labels, print_details=True):
    pred = model.predict(data)
    probs = model.predict_proba(data)
    print('Predictions:', pred)
    print('Probabilies:', probs)
    # if print_details:
    #     print('score', accuracy_score(pred, labels))
    #     print('pred', pred)
    #     print('actual', labels)
    #     print(confusion_matrix(labels,pred))

    return pred


def get_prediction_and_prob(model, data):
    pred = model.predict(data)
    probs = model.predict_proba(data)
    return pred, probs


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


def generate_and_write_results_hard_voting(pro_data, model_gender, model_msi, gender_labels, msi_labels, sample_names, strict=True):
    gender_predictions = hv.hard_vote(model_gender, pro_data, gender_labels, 'gender')

    msi_predictions = hv.hard_vote(model_msi, pro_data, msi_labels, 'msi')
    outfile = open('subchallenge_1.csv','w')
    outfile.write('sample,mismatch\n')

    if strict:
        for i in range(0,len(msi_labels)):
            outfile.write(sample_names[i] + ',')
            if gender_labels[i] == gender_predictions[i] and msi_labels[i] == msi_predictions[i]:
                outfile.write('0\n')
            else:
                outfile.write('1\n')
    else:
        for i in range(0, len(msi_labels)):
            outfile.write(sample_names[i] + ',')
            if gender_labels[i] != gender_predictions[i] and msi_labels[i] != msi_predictions[i]:
                outfile.write('1\n')
            else:
                outfile.write('0\n')
    outfile.close()

def generate_and_write_probability_voting(pro_data, model_gender, model_msi, gender_labels, msi_labels, sample_names, strict=True):
    gender_predictions = hv.hard_vote(model_gender, pro_data, gender_labels, 'gender')

    msi_predictions = hv.hard_vote(model_msi, pro_data, msi_labels, 'msi')
    outfile = open('subchallenge_1.csv','w')
    outfile.write('sample,mismatch\n')

    if strict:
        for i in range(0,len(msi_labels)):
            outfile.write(sample_names[i] + ',')
            if gender_labels[i] == gender_predictions[i] and msi_labels[i] == msi_predictions[i]:
                outfile.write('0\n')
            else:
                outfile.write('1\n')
    else:
        for i in range(0, len(msi_labels)):
            outfile.write(sample_names[i] + ',')
            if gender_labels[i] != gender_predictions[i] and msi_labels[i] != msi_predictions[i]:
                outfile.write('1\n')
            else:
                outfile.write('0\n')
    outfile.close()
