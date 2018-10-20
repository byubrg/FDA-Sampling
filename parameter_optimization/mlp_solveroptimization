from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
import pandas as pd
import seaborn as sns
import imp

RAND_STATE = 0
TEST_SIZE = 0.3
NUMBER_OF_SPLITS = 10
SCORING_METHOD = 'accuracy'

cv = StratifiedShuffleSplit(
        n_splits=NUMBER_OF_SPLITS,
        test_size=TEST_SIZE,
        random_state=RAND_STATE)


def train_mlp(data, labels):

    mlp = MLPClassifier(solver='sgd', alpha=1e-5, random_state=1, max_iter=200)
    mlp.fit(data, labels)

    scores = cross_val_score(mlp, data, labels, cv=cv, scoring=SCORING_METHOD)

    print(scores)


# read in the training files
clinical = pd.read_csv('/Users/DallasLarsen/Desktop/BRG Stuff/FDA_Sampling/Data/data/raw/train_cli.tsv', sep='\t')
protein = pd.read_csv('/Users/DallasLarsen/Desktop/BRG Stuff/FDA_Sampling/Data/data/raw/train_pro.tsv', sep='\t')
# protein = pd.read_csv('/Users/DallasLarsen/Desktop/BRG Stuff/FDA_Sampling/Data/data/pro_data_normalized.tsv')

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



# Max iterations
# maxIters = range(195, 205)

solvers = ["lbfgs", "sgd"]

# final_scores is being initialized as a list that will contain the mean scores for each value of k
final_scores = []

#train_mlp(protein_data, combind_labels)

# loop over the various values of k that need to be tested
for solver in solvers:

    mlp = MLPClassifier(solver=solver, alpha=1e-5, random_state=1, max_iter=300)
    mlp.fit(protein_data, combind_labels)
    scores = cross_val_score(mlp, protein_data, combind_labels, cv=cv, scoring=SCORING_METHOD)

    # get the mean of the scores for this iteration of the algorithm
    mean = sum(scores) / float(len(scores))

    print(solver)
    # add the mean final_score list so we can compare and plot it later
    final_scores.append(mean)


# create a dictionary of the data we have gathered
data = {'solve_method': solvers, 'score': final_scores}

# covert the data into a pandas dataframe for easy use in plotting
df = pd.DataFrame(data)

# create the plot and label the plot
ax = sns.barplot(x="solve_method", y="score", data=df).set_title('MLP - Optimization of Solver')

# save the plot
ax.figure.savefig("/Users/DallasLarsen/Desktop/BRG Stuff/FDA_Sampling/MLP_MaxIter_3.png")
