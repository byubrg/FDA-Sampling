import pandas as pd

import feature_selection
from siamese_net import SiameseNet
from load_data import LoadData

path = "./data/tidy/output/pairwise_scores.csv"

def collapse_comparisons():
    prot = "Proteomic"
    rna = "RNA"
    prob = "Probability"
    df = pd.read_csv("data/tidy/output/pairwise_scores.csv")
    df['sorted'] = df.apply(lambda x: ",".join(sorted([x[prot], x[rna]])), axis=1)
    split = df.groupby(['sorted']).mean().reset_index()['sorted'].str.split(',', 1, expand=True)
    means = df.groupby(['sorted']).mean()
    split[prob] = means.reset_index()[prob]
    return split


if __name__ == "__main__":
    # Train siamese network
    data = LoadData()
    pro_data = feature_selection.univariate(data.proteomic, data.clinical)
    rna_data = feature_selection.univariate(data.rna, data.clinical)

    prot_x = pd.concat([pro_data, pro_data, pro_data])
    shuffled_rna = rna_data.sample(frac=1)
    rna_x = pd.concat([rna_data, shuffled_rna, rna_data.sample(frac=1)])
    labels = [1.0] * 80 + [0.0] * 160

    network = SiameseNet([
        (pro_data.shape[-1],),
        (rna_data.shape[-1],)
    ])
    network.fit([prot_x, rna_x], labels, epochs=20, batch_size=5)

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
    probs.to_csv(path)

    # Get lowest 10% of self-matching
    probs = collapse_comparisons()
    probs['samples'] = probs.apply(lambda x: set([x[0], x[1]]), axis=1)
    self_matches = probs[probs[0] == probs[1]]
    bad_matches = self_matches.sort_values(["Probability"]).head(8)
    def adjust_probability(row):
        if row[0] == row[1]:
            return row["Probability"]
        prob_0_0 = probs[(probs[0] == row[0]) & (probs[1] == row[0])]['Probability'].iloc[0]
        prob_1_1 = probs[(probs[0] == row[1]) & (probs[1] == row[1])]['Probability'].iloc[0]
        return row["Probability"] - prob_0_0 - prob_1_1
    probs['adj_prob'] = probs.apply(adjust_probability, axis=1)
    print(probs[probs['adj_prob'] > 0])

    # For bottom 10%, find best matches

    # Output new matches: [[1,1], [2,3], [3,2], [4,4]]
