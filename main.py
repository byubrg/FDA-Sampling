import learner_functions as lf
import load_data as ld
import hard_vote as hv
import soft_vote as sv

data = ld.LoadData()

#create test and train labels
gender_labels = data.clinical['gender'].tolist()
MSI_labels = data.clinical['msi'].tolist()
test_gender_labels = data.test_clinical['gender'].tolist()
test_MSI_labels = data.test_clinical['msi'].tolist()

#train learners for gender and msi here:
knn_gender, knn_gender_score = lf.train_knn(data.proteomic,gender_labels)
knn_msi, knn_msi_score = lf.train_knn(data.proteomic,MSI_labels)
lr_gender, lr_gender_score = lf.train_lr(data.proteomic,gender_labels)
lr_msi, lr_msi_score = lf.train_lr(data.proteomic,MSI_labels)
rf_gender, rf_gender_score = lf.train_rf(data.proteomic,gender_labels)
rf_msi, rf_msi_score = lf.train_rf(data.proteomic,MSI_labels)
sgd_gender, sgd_gender_score = lf.train_sgd(data.proteomic,gender_labels)
sgd_msi, sgd_msi_score = lf.train_sgd(data.proteomic,MSI_labels)
nc_gender, nc_gender_score = lf.train_nc(data.proteomic,gender_labels)
nc_msi, nc_msi_score = lf.train_nc(data.proteomic,MSI_labels)
mlp_gender, mlp_gender_score = lf.train_mlp(data.proteomic,gender_labels)
mlp_msi, mlp_msi_score = lf.train_mlp(data.proteomic,MSI_labels)

#make final predictions here, give it the two trained classifiers
lf.generate_and_write_results(data.test_proteomic.fillna(0.0),
                              knn_gender,
                              knn_msi,
                              test_gender_labels,
                              test_MSI_labels,
                              list(data.test_proteomic.index))


#Not sure if this should be here in the main.

#All of the other models that are not being trained here for gender right now should be added to this array.
modelArrayGen = [knn_gender, lr_gender, rf_gender, sgd_gender, nc_gender, mlp_gender]
print("\n\n******************************************************************\nHARD VOTE FOR GENDERS")
hvGender = hv.hard_vote(modelArrayGen, data.proteomic, gender_labels, 'gender')
print(hvGender)
print("******************************************************************")

#All of the other models that are not being trained here for msi right now should be added to this array.
print("\n\n******************************************************************\nHARD VOTE FOR MSI")
modelArrayMSI = [knn_msi, lr_msi, rf_msi, sgd_msi, nc_msi, mlp_msi]
hvMSI = hv.hard_vote(modelArrayMSI, data.proteomic, lr_msi, 'msi')
print(hvMSI)
print("******************************************************************")

# Soft Voting
print("\n\n******************************************************************\n")
print("SOFT VOTE FOR GENDERS")
gender_estimators = [('knn', knn_gender), ('lr', lr_gender), ('rf', rf_gender), ('sgd', sgd_gender), ('nc', nc_gender), ('mlp', mlp_gender)]
sv_gender = sv.soft_vote(gender_estimators, data.proteomic, gender_labels)
print("******************************************************************")

print("\n\n******************************************************************\n")
print("SOFT VOTE FOR MSI")
msi_estimators = [('knn', knn_msi), ('lr', lr_msi), ('rf', rf_msi), ('sgd', sgd_msi), ('nc', nc_msi), ('mlp', mlp_msi)]
sv_msi = sv.soft_vote(msi_estimators, data.proteomic, MSI_labels)
print("******************************************************************")
