import learner_functions as lf
import load_data as ld
import subchallenge2 as sb


if __name__ == "__main__":
    data = ld.LoadData()
    pro_mms, rna_mms, cli_mms = sb.one_of_these_is_not_like_the_other(data.test_pro_rna.fillna(0), data.test_pro_cli.fillna(0), data.test_rna_cli.fillna(0), data.train_all.fillna(0), data.mismatch.fillna(0))
    print("Pro")
    print(pro_mms)
    print("rna")
    print(rna_mms)
    print("Clinical")
    print(cli_mms)
