import learner_functions as lf
import load_data as ld

data = ld.LoadData()

#create test and train labels
gender_labels = data.clinical['gender'].tolist()
MSI_labels = data.clinical['msi'].tolist()
test_gender_labels = data.test_clinical['gender'].tolist()
test_MSI_labels = data.test_clinical['msi'].tolist()

filtered_protein = data.proteomic[["ARHGAP6","EIF1AY","RPS4Y1","RPS4Y2","SRPK3","STS","UPRT","ZNF280C"]]
filtered_protein = data.proteomic
#train learners for gender and msi here:
knn_gender, a = lf.train_svm(filtered_protein,gender_labels)
knn_gender, b = lf.train_rf(filtered_protein,gender_labels)
knn_gender, c = lf.train_lr(filtered_protein,gender_labels)
knn_gender, d = lf.train_sgd(filtered_protein,gender_labels)
knn_gender, e = lf.train_nc(filtered_protein,gender_labels)
#knn_gender, f = lf.train_mlp(filtered_protein,gender_labels)
scores = [a,b,c,d,e]
for score in scores:
    print(score)
#knn_msi = lf.train_knn(data.proteomic.iloc[:,1:],MSI_labels)



#make final predictions here, give it the two trained classifiers
"""
lf.generate_and_write_results(data.test_proteomic.fillna(0.0).iloc[:,1:],
                              knn_gender,
                              knn_msi,
                              test_gender_labels,
                              test_MSI_labels,
                              list(data.test_proteomic.index))

Filtered Gender Scores
0.73125
0.83125
0.7725
0.775
0.835
0.635
"""