from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
import pandas as pd
import seaborn as sns

RAND_STATE = 0
TEST_SIZE = 0.3
NUMBER_OF_SPLITS = 10
SCORING_METHOD = 'accuracy'

cv = StratifiedShuffleSplit(
        n_splits=NUMBER_OF_SPLITS,
        test_size=TEST_SIZE,
        random_state=RAND_STATE)


def train_sgd(data, labels):

    sgd = SGDClassifier(shuffle=True)
    sgd.fit(data, labels)

    scores = cross_val_score(sgd, data, labels, cv=cv, scoring=SCORING_METHOD)

    print(scores)


# read in the training files
clinical = pd.read_csv('..data/raw/train_cli.tsv', sep='\t')
protein = pd.read_csv('..data/raw/train_pro.tsv', sep='\t')

# impute missing values with 0
protein = protein.fillna(0)
clinical = clinical.fillna(0)

# remove the sample labels and transform the data such that proteins are features
protein_data = protein.iloc[:,1:].T

# extract single columns of information and convert to a list
gender_labels = clinical['gender'].tolist()
MSI_labels = clinical['msi'].tolist()

# combind the gender and msi columns into one
clinical['combind'] = clinical['gender'] + clinical['msi']
combind_labels = clinical['combind'].tolist()

# raw
# [0.48275862 0.5        0.28      ]   test1
# [0.5        0.29166667 0.58333333 0.29166667 0.54166667 0.20833333  0.29166667 0.5        0.54166667 0.54166667]

# normalized
# [0.125      0.375      0.5        0.29166667 0.54166667 0.41666667   0.125      0.16666667 0.41666667 0.5       ]
# [0.5        0.16666667 0.375      0.54166667 0.625      0.5          0.54166667 0.54166667 0.29166667 0.625     ]
# [0.33333333 0.625      0.5        0.29166667 0.45833333 0.16666667   0.5        0.58333333 0.45833333 0.5       ]
# [0.41666667 0.16666667 0.54166667 0.58333333 0.375      0.29166667   0.5        0.58333333 0.41666667 0.29166667]

# Max iterations
maxIters = range(1, 11)

# final_scores is being initialized as a list that will contain the mean scores for each value of k
final_scores = []



# train_sgd(protein_data, combind_labels)

# loop over the various values of k that need to be tested
for currIter in maxIters:

    sgd = SGDClassifier(shuffle=True, max_iter=currIter)
    sgd.fit(protein_data, combind_labels)
    scores = cross_val_score(sgd, protein_data, combind_labels, cv=cv, scoring=SCORING_METHOD)

    # get the mean of the scores for this iteration of the algorithm
    mean = sum(scores) / float(len(scores))

    # add the mean final_score list so we can compare and plot it later
    final_scores.append(mean)

# create a dictionary of the data we have gathered
data = {'max_iterations': maxIters, 'score': final_scores}

# covert the data into a pandas dataframe for easy use in plotting
df = pd.DataFrame(data)

# create the plot and label the plot
ax = sns.barplot(x="max_iterations", y="score", data=df).set_title('SGD - Optimization of Max Iterations')

# save the plot
ax.figure.savefig("sgd-max-iterations-optimization-plot.png")
