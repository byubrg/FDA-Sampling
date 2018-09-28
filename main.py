import Learner_Functions as fl
import pandas as pd

#read in the training files
clinical = pd.read_csv('train_cli.tsv', sep='\t')
protein = pd.read_csv('train_pro.tsv', sep='\t')

#impute missing values with 0
protein = protein.fillna(0)
clinical = clinical.fillna(0)

#remove the sample labels and transform the data such that protiens are features
protein_data = protein.iloc[:,1:].T
gender_labels = clinical.iloc[:,1:2]
MSI_labels = clinical.iloc[:,2:1]


fl.train_knn(protein_data,gender_labels)
