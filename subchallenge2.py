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
mismatch_labels = data.mismatch['mismatch'].tolist()

# train models for predicting mislabels based on all 3 data sets
lf.train_rf(data.train_all.fillna(0), data.mislabel_labels)

# do deconvolution
