"""
Contained in file are functions used for 
    - training scikit learn classifiers
    - making prediction with each algorithm
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn import svm
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import pandas as pd
import seaborn as sns

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

def train_lr(data, labels):
    train_classifier(data, labels, LogisticRegression)

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

def svm_parameters(data, labels):
	#Kernel Poly
	kernelScores = []
	cValues = []
	for i in range(10):
		cValues.append(i)
		SVM = svm.SVC(kernel = 'linear', gamma = "scale", C = i)
		score = train_svm(data, labels, SVM)
		kernelScores.append(score)
	data = {'C': cValues, 'Scores':kernelScores}
	df = pd.DataFrame(data)
	ax = sns.barplot(x = 'C', y = 'Scores', data = df).set_title('Linear Kernel Optimization')
	ax.figure.savefig('LinearKernelOptimization.png')

def train_svm(data, labels, SVM):
	cv = StratifiedShuffleSplit(n_splits = NUMBER_OF_SPLITS, test_size = TEST_SIZE, random_state = RAND_STATE)
	scores = cross_val_score(SVM, data, labels, cv = cv, scoring = SCORING_METHOD)
	avg = sum(scores) / float(len(scores))
	return avg

#	parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
#	svc = svm.SVC(gamma="scale")
#	clf = GridSearchCV(svc, parameters, cv=cv)
#	clf.fit(data, labels)


