import learner_functions as lf
import load_data as ld
import feature_selection as fs
import hard_vote as hv
import soft_vote as sv

data = ld.LoadData()

#create test and train labels
gender_labels = data.clinical['gender'].tolist()
MSI_labels = data.clinical['msi'].tolist()
test_gender_labels = data.test_clinical['gender'].tolist()
test_MSI_labels = data.test_clinical['msi'].tolist()

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

print('sgd')
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

print('mlp')#optimization has been hardcoded in
mlp_gender, mlp_gender_score = lf.train_mlp(data.proteomic,gender_labels)
mlp_msi, mlp_msi_score = lf.train_mlp(data.proteomic,MSI_labels)

modelArrayGen = [knn_gender, lr_gender, rf_gender, nc_gender, svm_gender]
modelArrayMSI = [knn_msi, lr_msi, rf_msi, nc_msi, svm_msi]

modelArrayGen = [knn_gender, lr_gender, rf_gender, nc_gender, svm_gender, mlp_gender, sgd_gender]
modelArrayMSI = [knn_msi, lr_msi, rf_msi, nc_msi, svm_msi, mlp_msi, sgd_msi]


#make final predictions here, give it the two trained classifiers
lf.generate_and_write_results_hard_voting(test_protein_sub_set,
                                          modelArrayGen,
                                          modelArrayMSI,
                                          test_gender_labels,
                                          test_MSI_labels,
                                          list(data.test_proteomic.index),
                                          True)
