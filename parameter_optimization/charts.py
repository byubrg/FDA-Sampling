import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/cool_output_stuff.csv")

pd.set_option("display.max_columns", 500)

sns.set()
plt.ylim(0.45,0.52)
sns.barplot(x="param_n_estimators", y="mean_test_score", data=df)
sns.barplot(x="param_criterion", y="mean_test_score", data=df)
sns.barplot(x="param_criterion", y="mean_test_score", data=df)
plt.show()
