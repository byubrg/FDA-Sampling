import learner_functions as lf
import pandas as pd

#read in the training files
clinical = pd.read_csv('data/raw/train_cli.tsv', sep='\t')
protein = pd.read_csv('data/raw/train_pro.tsv', sep='\t').T
labels = pd.read_csv('data/tidy/sum_tab_1.csv', sep=',')

#create the labels for which samples have been mislabeled
mismatch_labels = labels.mismatch.tolist()

#make the row names of clinical data equal to the first column's content
clinical.index = clinical['sample'].tolist()

#remove the first column
clinical = clinical.iloc[:,1:]

#create column headers for protein data
protein.columns = protein.iloc[0]
protein = protein.iloc[1:]

#impute missing values with 0
protein = protein.fillna(0)
clinical = clinical.fillna(0)

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

#combined the gender and msi columns into one
clinical['combined'] = clinical['gender'] + clinical['msi']
combined_labels = clinical['combined'].tolist()

lf.train_rf(joint_data,mismatch_labels)
lf.train_knn(joint_data,mismatch_labels)
lf.train_sgd(joint_data,mismatch_labels)
lf.train_nc(joint_data,mismatch_labels)
lf.train_bagging_knn(joint_data,mismatch_labels)
