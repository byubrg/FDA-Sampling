import learner_functions as lf
import load_data as ld
import feature_selection as fs
import pandas as pd

data = ld.LoadData()

#create test and train labels
gender_labels = data.clinical['gender'].tolist()
MSI_labels = data.clinical['msi'].tolist()
test_gender_labels = data.test_clinical['gender'].tolist()
test_MSI_labels = data.test_clinical['msi'].tolist()

# Use feature selection on the data, get the same subset of features for the test data too
protein_sub_set = fs.univariate(data.proteomic, gender_labels)
selected_protein_columns = list(protein_sub_set.columns.values)
test_protein_sub_set = data.test_proteomic.ix[:, selected_protein_columns]

print('knn')
knn_params = { # Found by parameter optimization in knn-optimization.py
    "n_neighbors": 11
}

# train learners for gender and msi here:
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

print('NC Euclid')
nc_param = {
    'metric': 'euclidean'
}
nc_gender, nc_gender_score = lf.train_nc(protein_sub_set,gender_labels, **nc_param)
nc_msi, nc_msi_score = lf.train_nc(protein_sub_set,MSI_labels, **nc_param)

print('SVM linear')
svm_param = {
    'kernel': 'linear',
    'probability': True,
}
svm_gender, svm_gender_score = lf.train_svm(protein_sub_set,gender_labels,**svm_param)
svm_msi, svm_msi_score = lf.train_svm(protein_sub_set,MSI_labels,**svm_param)

print('mlp')#optimization has been hardcoded in
mlp_gender, mlp_gender_score = lf.train_mlp(protein_sub_set,gender_labels)
mlp_msi, mlp_msi_score = lf.train_mlp(protein_sub_set,MSI_labels)

print('SGD')
sgd_gender, sgd_gender_score = lf.train_sgd(protein_sub_set,gender_labels)
sgd_msi, sgd_msi_score = lf.train_sgd(protein_sub_set,MSI_labels)

modelArrayGen = [knn_gender, lr_gender, rf_gender, svm_gender, mlp_gender]
modelArrayMSI = [knn_msi, lr_msi, rf_msi, svm_msi, mlp_msi]

"""
give a group of trained models, test data and test labels
returns array of 0s and 1s, 1 indicating a mismatch
"""
def prob_based_mismatches(models, data, labels):
    # DataFrame for storing the mismatch predictions of each model
    # predefine the number of columns as the number of models
    results = pd.DataFrame(columns=range(0, len(models)))
    model_count = 0
    for model in models:
        pred, prob = lf.get_prediction_and_prob(model, data)
        count = 0
        count_confidant = 0
        mismatches = []
        for p in range(0, len(pred)):
            # if the predicted and actual label don't match
            if pred[p] != labels[p]:
                count += 1
                # if the prediction is above 75% confidant on it's classification
                if prob[p][0] > .8 or prob[p][1] > .8:
                    count_confidant += 1
                    # mark this sample as mislabeled
                    mismatches.append(1)
                else:
                    # mark as not mislabeled
                    mismatches.append(0)
            else:
                # mark as not mislabeled
                mismatches.append(0)
        # add the current preditions to the data frame containing all model's predictions
        results[model_count] = mismatches
        model_count += 1

    consensus = []

    # get the consensus of all the models
    for i in range(0, len(results.index)):
        total = sum(results.iloc[i, :])
        # if the number that vote mismatched are greater than half the total votes
        if total > (float(len(results.columns)) / 2.0):
            # mark as mismatched
            consensus.append(1)
        else:
            # mark as not mismatched
            consensus.append(0)
    print(consensus)
    print(sum(consensus))
    return consensus


gender_mismatch_predictions = prob_based_mismatches(modelArrayGen,test_protein_sub_set,test_gender_labels)
msi_mismatch_prediction = prob_based_mismatches(modelArrayMSI,test_protein_sub_set,test_MSI_labels)

outfile = open('subchallenge_1.csv','w')
outfile.write('sample,mismatch\n')

sample_names = list(data.test_proteomic.index)
count = 0

for i in range(0, len(gender_mismatch_predictions)):
    outfile.write(sample_names[i] + ',')
    # if either the gender or msi are considered mismatched
    if gender_mismatch_predictions[i] == 1 or msi_mismatch_prediction[i] == 1:
        outfile.write('1\n')
        count += 1
    else:
        outfile.write('0\n')

outfile.close()

# count is the total number of samples that are being labeled as mislabeled
print(count)




