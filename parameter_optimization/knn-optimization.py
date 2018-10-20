import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
import seaborn as sns

RAND_STATE = 0
TEST_SIZE = 0.3
NUMBER_OF_SPLITS = 10
SCORING_METHOD = 'accuracy'

#read in the training files
clinical = pd.read_csv('../data/raw/train_cli.tsv', sep='\t')
protein = pd.read_csv('../data/raw/train_pro.tsv', sep='\t')

#impute missing values with 0
protein = protein.fillna(0)
clinical = clinical.fillna(0)

#remove the sample labels and transform the data such that proteins are features
protein_data = protein.iloc[:,1:].T

#extract single columns of information and convert to a list
gender_labels = clinical['gender'].tolist()
MSI_labels = clinical['msi'].tolist()

#combind the gender and msi columns into one
clinical['combind'] = clinical['gender'] + clinical['msi']
combind_labels = clinical['combind'].tolist()

cv = StratifiedShuffleSplit( 
    n_splits = NUMBER_OF_SPLITS,
    test_size = TEST_SIZE,
    random_state = RAND_STATE
)

#-----below here is unique to knn------ 

#ks is the values of k that need to be evaluated for parameter optimization
ks = [1,3,5,7,9,11,13,15,17,19,21]

#final_scores is being initialized as a list that will contain the mean scores for each value of k
final_scores = []

#loop over the various values of k that need to be tested
for k in ks:
    #define a knn object with the parameter n_neighbors set to k
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, protein_data, combind_labels, cv = cv, scoring = SCORING_METHOD)

    #get the mean of the scores for this iteration of the algorithsm
    mean = sum(scores) / float(len(scores))

    #add the mean final_score list so we can compare and plot it later
    final_scores.append(mean)

#create a dictionary of the data we have gathered
data = {'k':ks,'score':final_scores}

#covert the data into a pandas dataframe for easy use in plotting
df = pd.DataFrame(data)

#create the plot and label the plot
ax = sns.barplot(x="k", y="score", data=df).set_title('KNN n-neighbors Parameter')

#save the plot
ax.figure.savefig("knn-n-neighbor-optimization-plot.png")

