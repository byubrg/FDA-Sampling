import learner_functions as lf
import pandas as pd

#read in the training files
clinical = pd.read_csv('data/raw/train_cli.tsv', sep='\t')
protein = pd.read_csv('data/raw/train_pro.tsv', sep='\t').T
labels = pd.read_csv('data/tidy/sum_tab_1.csv', sep=',')

#read in the test data
test_cli = pd.read_csv('data/raw/test_cli.tsv', sep='\t')
test_pro = pd.read_csv('data/raw/test_pro.tsv', sep='\t').T

#create the labels for which samples have been mislabeled
mismatch_labels = labels.mismatch.tolist()

#make the row names of clinical data equal to the first column's content
clinical.index = clinical['sample'].tolist()
test_cli.index = test_cli['sample'].tolist()

#remove the first column
clinical = clinical.iloc[:,1:]
test_cli = test_cli.iloc[:,1:]

#create column headers for protein data
protein.columns = protein.iloc[0]
protein = protein.iloc[1:]

test_pro.columns = test_pro.iloc[0]
#test_pro = test_pro.iloc[1:]

#impute missing values with 0
protein = protein.fillna(0)
clinical = clinical.fillna(0)
test_pro = test_pro.fillna(0)
test_cli = test_cli.fillna(0)

#combined the clinical and the protein data
joint_data = protein.combine_first(clinical)

#replace the nominal values from the clinical set with continuous values
joint_data = joint_data.replace('MSI-Low/MSS',0)
joint_data = joint_data.replace('MSI-High',1)
joint_data = joint_data.replace('Male',1)
joint_data = joint_data.replace('Female',0)

#extract single columns of information and convert to a list
gender_labels = clinical['gender'].tolist()
MSI_labels = clinical['msi'].tolist()

#get the gender and msi labels for test data
test_gender_labels = test_cli['gender'].tolist()
test_MSI_labels = test_cli['msi'].tolist()

#combined the gender and msi columns into one
clinical['combined'] = clinical['gender'] + clinical['msi']
combined_labels = clinical['combined'].tolist()

#train a learner
knn_gender = lf.train_knn(protein,gender_labels)
knn_msi = lf.train_knn(protein,MSI_labels)


lf.make_test_prediction(knn_msi,test_pro,test_MSI_labels)
lf.make_test_prediction(knn_gender,test_pro,test_gender_labels)
lf.generate_and_write_results(test_pro,knn_gender,knn_msi,test_gender_labels,test_MSI_labels,list(test_cli.index))