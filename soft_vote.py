from sklearn.ensemble import VotingClassifier
import learner_functions as lf

def soft_vote(estimators, test_data, test_labels):
    eclf = VotingClassifier(estimators=estimators, voting='soft')
    eclf = eclf.fit(test_data, test_labels)
    pred = lf.make_test_prediction(eclf, test_data, test_labels, print_details=True)
    return pred
    