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

def clinical_labels_dict():
    data = LoadData()

    def clinical_to_int(row):
        output = 0
        if row.msi == "MSI-High":
            output += 2
        if row.gender == "Female":
            output += 1
        return output

    labels = data.clinical.apply(clinical_to_int, axis="columns")
    return {sample: label for sample, label in zip(data.clinical.index, labels)}

def clinical_probabilities(learner=learner_functions.train_rf):
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

    model = learner(data.proteomic, labels)
    proteomic_probabilities_df = pd.DataFrame(model[0].predict_proba(data.proteomic))
    proteomic_probabilities_df['sample'] = data.clinical.index
    proteomic_probabilities_df = proteomic_probabilities_df.set_index('sample')

    model = learner(data.rna, labels)
    rna_probabilities_df = pd.DataFrame(model[0].predict_proba(data.rna))
    rna_probabilities_df['sample'] = data.clinical.index
    rna_probabilities_df = rna_probabilities_df.set_index('sample')

    return proteomic_probabilities_df, rna_probabilities_df

def rna_proteomic_mismatch_probabilities():
    # Train siamese network
    data = LoadData()
    pro_data = data.proteomic
    rna_data = data.rna

    prot_x = pd.concat([pro_data, pro_data, pro_data])
    shuffled_rna = rna_data.sample(frac=1)
    rna_x = pd.concat([rna_data, shuffled_rna, rna_data.sample(frac=1)])
    labels = [1.0] * 80 + [0.0] * 160

    network = SiameseNet([
        (pro_data.shape[-1],),
        (rna_data.shape[-1],)
    ])
    network.fit([prot_x, rna_x], labels, epochs=100, batch_size=5, verbose=False)

    # Calculate pairwise probabilities
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
    order = data.clinical.index.tolist()
    probs = probs.pivot(index='RNA', columns='Proteomic', values='Probability')[order].reindex(order)
    return probs

if __name__ == "__main__":
    prot_clinical, rna_clinical = clinical_probabilities()
    rna_prot = rna_proteomic_mismatch_probabilities()
    prot_clinical.to_csv("./data/probabilities/clinical_proteomic.csv")
    rna_clinical.to_csv("./data/probabilities/clinical_rna.csv")
    rna_prot.to_csv("./data/probabilities/rna_proteomic.csv")
