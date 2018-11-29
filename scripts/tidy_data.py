import pandas as pd

# Proteomic data
pd.read_table("data/raw/train_pro.tsv", index_col=0).T.reset_index().rename(columns={"index": "sample"}).to_csv("data/tidy/train_pro.csv", index=False)
# RNA data
pd.read_table("data/raw/train_rna.tsv").T.reset_index().rename(columns={"index": "sample"}).to_csv("data/tidy/train_rna.csv", index=False)
# Clinical data
pd.read_table("data/raw/train_cli.tsv").to_csv("data/tidy/train_cli.csv", index=False)
