import learner_functions as lf
import load_data as ld
import feature_selection as fs
import find_mismatch as fm
import pandas as pd

data = ld.LoadData()

#create test and train labels
gender_labels = data.clinical['gender'].tolist()
MSI_labels = data.clinical['msi'].tolist()
test_gender_labels = data.test_clinical['gender'].tolist()
test_MSI_labels = data.test_clinical['msi'].tolist()
mismatch_labels = data.mismatch['mismatch'].tolist()

protein_sub_set = fs.univariate(data.proteomic, mismatch_labels)
rna_sub_set = fs.univariate(data.rna, mismatch_labels)


knn_params = { # Found by parameter optimization in knn-optimization.py
    "n_neighbors": 11
}

# train models for predicting mislabels based on all 3 data sets
lf.train_rf(data.train_all.fillna(0), data.mislabel_labels)

# ************************************
# TESTING TO FIND MISMATCH INDICES
# knn_rna, knn_rna_score = lf.train_knn(rna_sub_set, mismatch_labels, **knn_params)
# knn_protein, knn_protein_score = lf.train_knn(protein_sub_set, mismatch_labels, **knn_params)
#
# lr_rna, lr_rna_score = lf.train_lr(rna_sub_set,mismatch_labels)
# lr_msi, lr_msi_score = lf.train_lr(protein_sub_set,MSI_labels)
#
# lf.make_test_prediction(knn_rna, rna_sub_set, True)
#
# modelArray = [knn_rna, lr_rna]
# scoreList = [knn_rna_score, lr_rna_score]
# print("MISMATCH INDICIES")
# mismatches = fm.find_mismatch_indices_hard(modelArray, rna_sub_set, mismatch_labels)
# print(mismatches)
# print("FJIOEWAIO;JJFJIO;EAWJIOFEWAJIO;J")
# print("AJIPOEFWJIO;EAWJIFIJEAW;IOIJOFEWAJIOAWEIJFO;AWJEIO")
# print(fm.find_mismatch_probabilities(mismatches, modelArray, scoreList, rna_sub_set, mismatch_labels))

# END OF TESTING MISMATCH INDICES
# ************************************

# do feature selection
# create a single function that returns just the most important features from protein, rna and clinical data sets
#

# get mislabeled samples
# create a function to get list of samples that have mislabels return list of row indexes


# do deconvolution
# given a list of row indexes of samples that are mislabeled, correct the mislabeling
#   find which of the 3 types of data does not belong, rna, protein or clinical
#   brute force all combinations of similarly mismatched data do find best match
#   generate report
