# FDA-Sampling

Code for the [precisionFDA sample mislabeling identification challenge](https://precision.fda.gov/challenges/4).

## File Structure

### `data/`

This directory houses the all data used by this code for the challenge.

### `script/`

Contains stand allow scripts used for some functional axiliary purpose to the project. Such as cleaning the raw input data and creating tidy data.

### `r/`

Contains all R scripts and R markdown documents used in the analysis

### `learner_functions.py`

Is a python module intened to be used by other scripts and never to be run on it own.

Contained in file are functions used for 
    - training scikit learn classifiers
    - making prediction with each algorithm

To contribute a new model to this module do the following:

1. Add an import stament for just module being used.


```python
from sklearn.neighbors import KNeighborsClassifier
```

2. Create a function called train_name-of-model that takes two parameters, training data and labels for that data. This function should create a new classifier, make a stratified shuffled split of the data, get cross validation scores of the module and print the scores.


```python
def train_knn(data,labels):
    knn = KNeighborsClassifier()
    cv = StratifiedShuffleSplit( 
            n_splits = NUMBER_OF_SPLITS, 
            test_size = TEST_SIZE, 
            random_state = RAND_STATE )
    
    scores = cross_val_score(knn, data, labels, cv = cv, scoring = SCORING_METHOD)
    print(scores)
```

### `main.py`

This is the main script for executing the analysis and algorithm. It is from this file that learner_functions should be used.

## Feature Selection

There are three types of feature selection implemented:

* [Variance Threshold](http://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance) - Remove features with low variance.
* [Univariate Feature Selection](http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection) - Select the best features based on some metric. Default is `SelectKBest`, which gets the k features that classify with the highest score (default is accuracy).
* [Recursive Feature Elimination](http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection) - Recursively eliminate features that look less important after classification.

```python
import feature_selection
from load_data import LoadData
from sklearn.svm import SVC

data = LoadData()

var_threshold = feature_selection.variance(data.proteomic, threshold=0.125)
k_best = feature_selection.univariate(data.proteomic, data.clinical, method=SelectKBest)
feature_elim = feature_selection.elimination(data.proteomic, data.clinical, SVC(), eliminator=RFE, n_features_to_select=15)
```
