import pandas as pd

# Proteomic data
pd.read_table("data/raw/train_pro.tsv").T.reset_index().rename(columns={"index": "sample"}).to_csv("data/tidy/train_pro.csv", index=False)
# Clinical data
pd.read_table("data/raw/train_cli.tsv").to_csv("data/tidy/train_cli.csv", index=False)
