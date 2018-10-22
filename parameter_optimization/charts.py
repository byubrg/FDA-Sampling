import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/cool_output_stuff.csv")

pd.set_option("display.max_columns", 500)

param_cols = ["param_n_estimators", "param_criterion", "param_min_samples_leaf", "param_min_samples_split"]
value_cols = ["mean_test_score", "mean_fit_time"]

# df = pd.melt(df, id_vars=value_cols, value_vars=param_cols, var_name="param")
# df = df.assign(param=lambda df_: df_.apply(lambda row: row.param[6:], axis=1))

print(df)

fig, axs = plt.subplots(ncols=2, nrows=2, sharey=True)

plt.ylim(0.45, 0.52)

sns.set()

sns.barplot(x="param_n_estimators", y="mean_test_score", data=df, ax=axs[0][0])
sns.barplot(x="param_criterion", y="mean_test_score", data=df, ax=axs[0][1])
sns.barplot(x="param_min_samples_leaf", y="mean_test_score", data=df, ax=axs[1][0])
sns.barplot(x="param_min_samples_split", y="mean_test_score", data=df, ax=axs[1][1])

plt.show()
