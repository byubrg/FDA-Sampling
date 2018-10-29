import learner_functions as lf
import load_data as ld
import feature_selection as fs

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
rf_params = { # Found by parameter optimization in randomforest.py
    "criterion": 'gini', 
    "min_samples_leaf": 1, 
    "min_samples_split": 5,
    "n_estimators": 100
}
#change data.proteomic to most important features
rf_gender, rf_gender_score = lf.train_rf(
    fs.univariate(data.proteomic, gender_labels),
    gender_labels,
    **rf_params
)
rf_msi, rf_msi_score = lf.train_rf(
    fs.univariate(data.proteomic, MSI_labels),
    MSI_labels,
    **rf_params
)

#make final predictions here, give it the two trained classifiers
lf.generate_and_write_results(data.test_proteomic.fillna(0.0),
                              knn_gender,
                              knn_msi,
                              test_gender_labels,
                              test_MSI_labels,
                              list(data.test_proteomic.index))
