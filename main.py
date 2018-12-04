import learner_functions as lf
import load_data as ld
import feature_selection as fs
import hard_vote as hv
import find_mismatch as fm
import soft_vote as sv

data = ld.LoadData()

#create test and train labels
gender_labels = data.clinical['gender'].tolist()
MSI_labels = data.clinical['msi'].tolist()
test_gender_labels = data.test_clinical['gender'].tolist()
test_MSI_labels = data.test_clinical['msi'].tolist()

#feature selections and subsetted data
protein_sub_set = fs.univariate(data.proteomic, gender_labels)
selected_protein_columns = list(protein_sub_set.columns.values)
test_protein_sub_set = data.test_proteomic.ix[:, selected_protein_columns]


print('knn')
knn_params = { # Found by parameter optimization in knn-optimization.py
    "n_neighbors": 11
}
#train learners for gender and msi here:
knn_gender, knn_gender_score = lf.train_knn(protein_sub_set,gender_labels, **knn_params)
knn_msi, knn_msi_score = lf.train_knn(protein_sub_set,MSI_labels, **knn_params)

print('lr')
lr_gender, lr_gender_score = lf.train_lr(protein_sub_set,gender_labels)
lr_msi, lr_msi_score = lf.train_lr(protein_sub_set,MSI_labels)

print('rf')
rf_params = { # Found by parameter optimization in randomforest.py
    "criterion": 'gini',
    "min_samples_leaf": 1,
    "min_samples_split": 5,
    "n_estimators": 100
}
#change data.proteomic to most important features
rf_gender, rf_gender_score = lf.train_rf(
    protein_sub_set,
    gender_labels,
    **rf_params
)
rf_msi, rf_msi_score = lf.train_rf(
    protein_sub_set,
    MSI_labels,
    **rf_params
)

print("sgd")
sgd_gender, sgd_gender_score = lf.train_sgd(data.proteomic,gender_labels)
sgd_msi, sgd_msi_score = lf.train_sgd(data.proteomic,MSI_labels)

print('NC Euclid')
nc_param = {
    'metric': 'euclidean'
}
nc_gender, nc_gender_score = lf.train_nc(protein_sub_set,gender_labels, **nc_param)
nc_msi, nc_msi_score = lf.train_nc(protein_sub_set,MSI_labels, **nc_param)

print('SVM linear')
svm_param = {
    'kernel': 'linear'
}
svm_gender, svm_gender_score = lf.train_svm(protein_sub_set,gender_labels,**svm_param)
svm_msi, svm_msi_score = lf.train_svm(protein_sub_set,MSI_labels,**svm_param)
#
print('mlp')#optimization has been hardcoded in
mlp_gender, mlp_gender_score = lf.train_mlp(data.proteomic,gender_labels)
mlp_msi, mlp_msi_score = lf.train_mlp(data.proteomic,MSI_labels)
#make final predictions here, give it the two trained classifiers
lf.generate_and_write_results(protein_sub_set,
                              knn_gender,
                              knn_msi,
                              test_gender_labels,
                              test_MSI_labels,
                              list(data.test_proteomic.index))


#Not sure if this should be here in the main.

#All of the other models that are not being trained here for gender right now should be added to this array.
modelArrayGen = [knn_gender, lr_gender, rf_gender, sgd_gender, nc_gender, mlp_gender]

print("\n\n******************************************************************\nHARD VOTE FOR GENDERS")
hvGender = hv.hard_vote(modelArrayGen, test_protein_sub_set, gender_labels, 'gender')
print(hvGender)
print("******************************************************************")

# #All of the other models that are not being trained here for msi right now should be added to this array.
print("\n\n******************************************************************\nHARD VOTE FOR MSI")
modelArrayMSI = [knn_msi, lr_msi, rf_msi, sgd_msi, nc_msi, mlp_msi]

hvMSI = hv.hard_vote(modelArrayMSI, test_protein_sub_set, lr_msi, 'msi')
print(hvMSI)
print("******************************************************************")


# Soft Voting

# NC does not have a 'predict_proba' attribute (i.e. no probability estimates)
# Probability estimates are not available for hinge loss in SGD. See learner_functions for a modified SGD trainer function.
# sgd_gender_mod, sgd_gender_score_mod = lf.train_sgd_mod(protein_sub_set,gender_labels)
# sgd_msi_mod, sgd_msi_score_mod = lf.train_sgd_mod(protein_sub_set,MSI_labels)
#
#
# print("\n\n******************************************************************")
# print("SOFT VOTE FOR GENDERS")
# gender_estimators = [('knn', knn_gender), ('lr', lr_gender), ('rf', rf_gender), ('sgd', sgd_gender_mod), ('mlp', mlp_gender)]
# sv_gender = sv.soft_vote(gender_estimators, data.proteomic, gender_labels)
# print("******************************************************************")
#
# print("\n\n******************************************************************")
# print("SOFT VOTE FOR MSI")
# msi_estimators = [('knn', knn_msi), ('lr', lr_msi), ('rf', rf_msi), ('sgd', sgd_msi_mod), ('mlp', mlp_msi)]
# sv_msi = sv.soft_vote(msi_estimators, data.proteomic, MSI_labels)
# print("******************************************************************")

mismatchTest = fm.find_all_mismatches(modelArrayGen, test_protein_sub_set)
print(mismatchTest)
