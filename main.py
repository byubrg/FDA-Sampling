import learner_functions as lf
import load_data as ld
import hard_vote as hv

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



#make final predictions here, give it the two trained classifiers
lf.generate_and_write_results(data.test_proteomic.fillna(0.0),
                              knn_gender,
                              knn_msi,
                              test_gender_labels,
                              test_MSI_labels,
                              list(data.test_proteomic.index))


#Not sure if this should be here in the main.

#All of the other models that are not being trained here for gender right now should be added to this array.
modelArrayGen = [knn_gender, lr_gender]
print("\n\n******************************************************************\nHARD VOTE FOR GENDERS")
hvGender = hv.hard_vote(modelArrayGen, data.proteomic, gender_labels, 'gender')
print(hvGender)
print("******************************************************************")

#All of the other models that are not being trained here for msi right now should be added to this array.
print("\n\n******************************************************************\nHARD VOTE FOR MSI")
modelArrayMSI = [knn_msi, lr_msi]
hvMSI = hv.hard_vote(modelArrayMSI, data.proteomic, lr_msi, 'msi')
print(hvMSI)
print("******************************************************************")
