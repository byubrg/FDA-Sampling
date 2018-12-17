import pandas as pd
from load_data import LoadData
from learner_functions import train_rf

CLINICAL_VALUES = {
    ("Female", "MSI-Low/MSS"): 0,
    ("Male",   "MSI-Low/MSS"): 1,
    ("Female", "MSI-High"):    2,
    ("Male",   "MSI-High"):    3,
}

def clinical_probabilities():
    """Get the probabilities of each clinical class for each sample for
    proteomic and rna data.

    Returns (proteomic_probabilities_df, rna_probabilities_df)
    """
    data = LoadData()

    def clinical_to_int(row):
        output = 0
        if row.msi == "MSI-High":
            output += 2
        if row.gender == "Female":
            output += 1
        return output

    labels = data.clinical.apply(clinical_to_int, axis="columns")

    rf = train_rf(data.proteomic, labels)
    proteomic_probabilities_df = pd.DataFrame(rf[0].predict_proba(data.proteomic))

    rf = train_rf(data.rna, labels)
    rna_probabilities_df = pd.DataFrame(rf[0].predict_proba(data.rna))

    return proteomic_probabilities_df, rna_probabilities_df
