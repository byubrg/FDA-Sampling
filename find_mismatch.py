import learner_functions as lf
import load_data as ld


def find_mismatch(model, data):

    return model.predict_proba(data)


def find_all_mismatches(models, data):

    probabilitiesOfModels = list()
    mismatchIndices = list()

    for model in models:
        probabilitiesOfModels.append(find_mismatch(model, data))

    for index in range(len(probabilitiesOfModels[0])):
        probabilitySum = 0
        for array in probabilitiesOfModels:
            probabilitySum += array[index]

        finalProbability = probabilitySum/len(probabilitiesOfModels)
        if finalProbability < 0.5:
            mismatchIndices.append(index)

    return mismatchIndices
