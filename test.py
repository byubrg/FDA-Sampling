import learner_functions as lf
import load_data as ld


if __name__ == "__main__":
    data = ld.LoadData()
    lf.train_rf(data.train_all.fillna(0), data.mislabel_labels)
    pro_vs_rna = lf.train_rf(data.train_pro_rna.fillna(0), data.mislabel_labels)
    pro_vs_cli = lf.train_rf(data.train_pro_cli.fillna(0), data.mislabel_labels)
    rna_vs_cli = lf.train_rf(data.train_rna_cli.fillna(0), data.mislabel_labels)
