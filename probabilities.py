import pandas as pd
from load_data import LoadData
import learner_functions
from siamese_net import SiameseNet

CLINICAL_VALUES = {
    ("Female", "MSI-Low/MSS"): 0,
    ("Male",   "MSI-Low/MSS"): 1,
    ("Female", "MSI-High"):    2,
    ("Male",   "MSI-High"):    3,
}

def clinical_labels_dict(train=True):
    data = LoadData()

    def clinical_to_int(row):
        output = 0
        if row.msi == "MSI-High":
            output += 2
        if row.gender == "Female":
            output += 1
        return output

    if train:
        clin_data = data.clinical
    else:
        clin_data = data.test_clinical

    labels = clin_data.apply(clinical_to_int, axis="columns")
    return {sample: label for sample, label in zip(clin_data.index, labels)}

def clinical_probabilities(train=True, learner=learner_functions.train_rf):
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

    train_labels = pd.concat([data.clinical, data.test_clinical]).apply(clinical_to_int, axis="columns")
    train_proteomic = pd.concat([data.proteomic, data.test_proteomic])
    train_rna = pd.concat([data.rna, data.test_rna])

    model = learner(train_proteomic, train_labels)
    if train:
        proteomic_probabilities_df = pd.DataFrame(model[0].predict_proba(data.proteomic))
        proteomic_probabilities_df['sample'] = data.clinical.index
    else:
        proteomic_probabilities_df = pd.DataFrame(model[0].predict_proba(data.test_proteomic))
        proteomic_probabilities_df['sample'] = data.test_clinical.index
    proteomic_probabilities_df = proteomic_probabilities_df.set_index('sample')

    model = learner(train_rna, train_labels)
    if train:
        rna_probabilities_df = pd.DataFrame(model[0].predict_proba(data.rna))
        rna_probabilities_df['sample'] = data.clinical.index
    else:
        rna_probabilities_df = pd.DataFrame(model[0].predict_proba(data.test_rna))
        rna_probabilities_df['sample'] = data.test_clinical.index
    rna_probabilities_df = rna_probabilities_df.set_index('sample')

    return proteomic_probabilities_df, rna_probabilities_df

def rna_proteomic_mismatch_probabilities(train=True):
    # Train siamese network
    data = LoadData()
    pro_data = data.proteomic
    rna_data = data.rna

    prot_x = pd.concat([pro_data]*3 + [data.test_proteomic]*3)
    shuffled_rna = rna_data.sample(frac=1)
    rna_x = pd.concat([rna_data, shuffled_rna, rna_data.sample(frac=1),
        data.test_rna, data.test_rna.sample(frac=1), data.test_rna.sample(frac=1)
    ])
    labels = [1.0] * 80 + [0.0] * 160 + [1.0] * 80 + [0.0] * 160

    network = SiameseNet([
        (pro_data.shape[-1],),
        (rna_data.shape[-1],)
    ])
    network.fit([prot_x, rna_x], labels, epochs=100, batch_size=5, verbose=False)

    # Calculate pairwise probabilities
    if not train:
        pro_data = data.test_proteomic
        rna_data = data.test_rna

    vals = {
        "Proteomic": [],
        "RNA": [],
        "Probability": [],
    }
    for i, x in pro_data.iterrows():
        for j, y in rna_data.iterrows():
            vals['Proteomic'].append(x.name)
            vals['RNA'].append(y.name)
            vals['Probability'].append(network.predict([[x], [y]])[0][0])

    probs = pd.DataFrame(vals)
    if train:
        order = data.clinical.index.tolist()
    else:
        order = data.test_clinical.index.tolist()
    probs = probs.pivot(index='RNA', columns='Proteomic', values='Probability')[order].reindex(order)
    return probs

if __name__ == "__main__":
    prot_clinical, rna_clinical = clinical_probabilities()
    rna_prot = rna_proteomic_mismatch_probabilities()
    prot_clinical.to_csv("./data/probabilities/clinical_proteomic.csv")
    rna_clinical.to_csv("./data/probabilities/clinical_rna.csv")
    rna_prot.to_csv("./data/probabilities/rna_proteomic.csv")

    prot_clinical_test, rna_clinical_test = clinical_probabilities(train=False)
    rna_prot_test = rna_proteomic_mismatch_probabilities(train=False)
    prot_clinical_test.to_csv("./data/probabilities/clinical_proteomic_test.csv")
    rna_clinical_test.to_csv("./data/probabilities/clinical_rna_test.csv")
    rna_prot_test.to_csv("./data/probabilities/rna_proteomic_test.csv")
