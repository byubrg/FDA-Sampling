# this uses a hard-vote-esque method to determine the mismatch (adds up number of mismatches (1s)
# then divides by number of models.  if less than 0.5, then it is not mismatch, and vice versa)

import learner_functions as lf
import load_data as ld

data = ld.LoadData()

mismatch_labels = data.mismatch['mismatch'].tolist()


def find_mismatch_indices_hard(models, data, labels, type="default"):

    predictionForEachModel = list()
    mismatchIndices = list()

    for model in models:
        predictionForEachModel.append(lf.make_test_prediction(model, data, labels, False))


    for index in range(len(predictionForEachModel[0])):
        predictionSum = 0
        for array in predictionForEachModel:
            predictionSum += array[index]

        finalPrediction = predictionSum / len(models)
        if(finalPrediction > 0.5):
            mismatchIndices.append(index)

    return mismatchIndices