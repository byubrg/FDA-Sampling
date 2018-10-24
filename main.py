import learner_functions as lf
import load_data as ld

data = ld.LoadData()

#create test and train labels
gender_labels = data.clinical['gender'].tolist()
MSI_labels = data.clinical['msi'].tolist()
test_gender_labels = data.test_clinical['gender'].tolist()
test_MSI_labels = data.test_clinical['msi'].tolist()

#train learners for gender and msi here:
knn_gender = lf.train_knn(data.proteomic,gender_labels)
knn_msi = lf.train_knn(data.proteomic,MSI_labels)
lr_gender = lf.train_lr(data.proteomic,gender_labels)
lr_msi = lf.train_lr(data.proteomic,MSI_labels)



#make final predictions here, give it the two trained classifiers
lf.generate_and_write_results(data.test_proteomic.fillna(0.0),
                              knn_gender,
                              knn_msi,
                              test_gender_labels,
                              test_MSI_labels,
                              list(data.test_proteomic.index))
