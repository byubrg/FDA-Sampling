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

print('SVM rbf')
svm_param = {
    'kernel': 'rbf'
}
svm_gender, svm_score = lf.train_svm(protein_sub_set,gender_labels,**svm_param)
svm_gender, svm_score = lf.train_svm(protein_sub_set,MSI_labels,**svm_param)
print('SVM linear')
svm_param = {
    'kernel': 'linear'
}
svm_gender, svm_score = lf.train_svm(protein_sub_set,gender_labels,**svm_param)
svm_gender, svm_score = lf.train_svm(protein_sub_set,MSI_labels,**svm_param)
print('SVM poly')
svm_param = {
    'kernel': 'poly'
}
svm_gender, svm_score = lf.train_svm(protein_sub_set,gender_labels,**svm_param)
svm_gender, svm_score = lf.train_svm(protein_sub_set,MSI_labels,**svm_param)
print('SVM sig')
svm_param = {
    'kernel': 'sigmoid'
}
svm_gender, svm_score = lf.train_svm(protein_sub_set,gender_labels,**svm_param)
svm_gender, svm_score = lf.train_svm(protein_sub_set,MSI_labels,**svm_param)
"""
#this one causes an error and is thus invalid for use, included here for documentation purposes
print('SVM precomp')
svm_param = {
    'kernel': 'precomputed'
}
svm_gender, svm_score = lf.train_svm(protein_sub_set,gender_labels,**svm_param)
svm_gender, svm_score = lf.train_svm(protein_sub_set,MSI_labels,**svm_param)
"""
print('SVM ovo')
svm_param = {
    'decision_function_shape': 'ovo'
}
svm_gender, svm_score = lf.train_svm(protein_sub_set,gender_labels,**svm_param)
svm_gender, svm_score = lf.train_svm(protein_sub_set,MSI_labels,**svm_param)

print('SVM ovr')
svm_param = {
    'decision_function_shape': 'ovr'
}
svm_gender, svm_score = lf.train_svm(protein_sub_set,gender_labels,**svm_param)
svm_gender, svm_score = lf.train_svm(protein_sub_set,MSI_labels,**svm_param)
