import learner_functions as lf
import load_data as ld


def hard_vote(models, data, labels, type):

    modelVotes = list()

    counter = 0
    for model in models:

        votes = lf.make_test_prediction(model, data, labels, False)

        modelVotes.append(votes.tolist())
        counter += 1

    finalVotes = list()

    if type == 'gender':
        for index in range(80):
            femaleCount = 0
            maleCount = 0
            for modelVoteArray in modelVotes:
                if modelVoteArray[index] == 'Female':
                    femaleCount += 1
                elif modelVoteArray[index] == 'Male':
                    maleCount += 1

            if femaleCount > maleCount:
                finalVotes.append('Female')
            else:
                finalVotes.append('Male')

    if type == 'msi':
        for index in range(80):
            msiHighCount = 0
            msiLowCount = 0
            for modelVoteArray in modelVotes:
                if modelVoteArray[index] == 'MSI-High':
                    msiHighCount += 1
                elif modelVoteArray[index] == 'MSI-Low/MSS':
                    msiLowCount += 1

            if msiHighCount > msiLowCount:
                finalVotes.append('MSI-High')
            else:
                finalVotes.append('MSI-Low/MSS')

    return finalVotes













