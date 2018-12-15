import learner_functions as lf
import load_data as ld
import feature_selection as fs
import pandas

# data = ld.LoadData()
#
# #create test and train labels
# gender_labels = data.clinical['gender'].tolist()
# MSI_labels = data.clinical['msi'].tolist()
# test_gender_labels = data.test_clinical['gender'].tolist()
# test_MSI_labels = data.test_clinical['msi'].tolist()
# mismatch_labels = data.mismatch['mismatch'].tolist()
#
# # train models for predicting mislabels based on all 3 data sets
# lf.train_rf(data.train_all.fillna(0), data.mislabel_labels)

# do feature selection
# create a single function that returns just the most important features from protein, rna and clinical data sets


# get mislabeled samples
# create a function to get list of samples that have mislabels return list of row indices
# Do we need to put these in as a object?
def one_of_these_is_not_like_the_other(test_pro_rna, test_pro_cli, test_rna_cli, training_df, training_labels):
    model = lf.train_one_of_these(training_df, training_labels)
    rna_mms = []
    pro_mms = []
    cli_mms = []

    # pro = []
    # rna = []
    # done = False
    # b = 1
    # c = 1
    # while not done:
    #     if test_pro[b, 0] == test_rna[c, 0]:
    #         pro.append(test_pro.ix[b, :])
    #         rna.append(test_rna.ix[c, :])
    #         b += 1
    #         c += 1
    #     if test_pro[b, 0] > test_rna[c, 0]:
    #         c += 1
    #     if test_pro[b, 0] < test_rna[c, 0]:
    #         b += 1
    #     if b == len(test_pro) or c == len(test_rna):
    #         done = True
    #     print(test_pro[b, 0])
    #     print(test_rna[c, 0])
    #     print(c)
    #     print(b)

    for a in range(1, 81):
        print(a)
        pro_mm_count = 0
        cli_mm_count = 0
        rna_mm_count = 0

        prm = lf.predict_one_of_these(model, test_pro_rna.ix[:, a])
        print("prm " + str(prm))
        if prm == 1:
            pro_mm_count += 1
            rna_mm_count += 1
        pcm = lf.predict_one_of_these(model, test_pro_cli.ix[:, a])
        print("pcm " + str(pcm))
        if pcm == 1:
            pro_mm_count += 1
            cli_mm_count += 1
        rcm = lf.predict_one_of_these(model, test_rna_cli.ix[:, a])
        print("rcm " + str(rcm))
        if rcm == 1:
            rna_mm_count += 1
            cli_mm_count += 1

        if pro_mm_count == 2:
            pro_mms.append(a)
            print("pro")
        elif rna_mm_count == 2:
            rna_mms.append(a)
            print("rna")
        else:
            cli_mms.append(a)
            print("Cli")

    return pro_mms, rna_mms, cli_mms



# do deconvolution
# given a list of row indexes of samples that are mislabeled, correct the mislabeling
#   find which of the 3 types of data does not belong, rna, protein or clinical
#   brute force all combinations of similarly mismatched data do find best match
#   generate report
