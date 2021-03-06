import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


class LoadData(object):
    def __init__(self,
                 clinical_path='data/tidy/train_cli.csv',
                 proteomic_path='data/tidy/train_pro.csv',
                 mismatch_path='data/tidy/sum_tab_1.csv'):
        self.clinical = pd.read_csv(clinical_path, index_col=0)
        self.proteomic = pd.read_csv(proteomic_path, index_col=0)
        self.proteomic = self.fix_data(self.proteomic)
        self.proteomic = self.normalize(self.proteomic)
        self.mismatch = pd.read_csv(mismatch_path, index_col=0)


    def normalize(self, df):
        return (df - df.mean()) / (df.max() - df.min())

    def fix_data(self, df):
        bad_columns = ["TMEM35A"]
        return df.fillna(0.0).drop(bad_columns, axis="columns")


if __name__ == "__main__":
    np.random.seed(0)
    data = LoadData()
    params = {
        "n_estimators": [50, 100, 150, 200],
        "criterion": ["gini", "entropy"],
        "min_samples_split": [2, 3, 4, 5],
        "min_samples_leaf": [1, 2, 3],
    }
    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, params)
    clf.fit(data.proteomic, pd.get_dummies(data.clinical))
    print("Best params:", clf.best_params_)
    df = pd.DataFrame(clf.cv_results_)

    # Make plots
    fig, axs = plt.subplots(ncols=2, nrows=2, sharey=True, figsize=(5, 6))
    plt.ylim(0.45, 0.52)
    plt.suptitle("Random Forests Parameter Optimization")
    val = "mean_test_score"
    # val = "mean_fit_time"
    sns.barplot(x="param_n_estimators", y=val, data=df, ax=axs[0][0])
    sns.barplot(x="param_criterion", y=val, data=df, ax=axs[0][1])
    sns.barplot(x="param_min_samples_leaf", y=val, data=df, ax=axs[1][0])
    sns.barplot(x="param_min_samples_split", y=val, data=df, ax=axs[1][1])

    fig.savefig("parameter_optimization/randomforest.png")
