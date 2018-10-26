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
import pandas as pd
import seaborn as sns

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
    return scores

def svmParameters(data, labels):
	#Kernel Poly
	kernelScores = []
	cValues = []
	for i in range(10):
		cValues.append(i)
		SVM = svm.SVC(kernel = 'linear', gamma = 'scale', C = 'i')
		score = train_classifier(data, labels, svm.SVC)
#(kernel = 'linear', gamma = 'scale', C = 'i'))
		kernelScores.append(score)
	data = {'C': cValues, 'Scores':kernelScores}
	df = pd.DataFrame(data)
	ax = sns.barplot(x = 'C', y = 'Scores', data = df).set_title('Linear Kernel Optimization')
	ax.firgure.savefig('LinearKernelOptimization.png')
