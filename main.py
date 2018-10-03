import learner_functions as lf
import pandas as pd

#read in the training files
clinical = pd.read_csv('data/raw/train_cli.tsv', sep='\t')
protein = pd.read_csv('data/raw/train_pro.tsv', sep='\t')

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

lf.train_knn(protein_data,combind_labels)
