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
    # # Train siamese network
    # data = LoadData()
    # pro_data = feature_selection.univariate(data.proteomic, data.clinical)
    # rna_data = feature_selection.univariate(data.rna, data.clinical)

    # prot_x = pd.concat([pro_data, pro_data])
    # shuffled_rna = rna_data.sample(frac=1)
    # rna_x = pd.concat([rna_data, shuffled_rna])
    # labels = [1.0] * 80 + [0.0] * 80

    # network = SiameseNet([
    #     (pro_data.shape[-1],),
    #     (rna_data.shape[-1],)
    # ])
    # network.fit([prot_x, rna_x], labels, epochs=20, batch_size=5)

    # # Calculate pairwise probabilities
    # vals = {
    #     "Proteomic": [],
    #     "RNA": [],
    #     "Probability": [],
    # }
    # for i, x in pro_data.iterrows():
    #     for j, y in rna_data.iterrows():
    #         vals['Proteomic'].append(x.name)
    #         vals['RNA'].append(y.name)
    #         vals['Probability'].append(network.predict([[x], [y]])[0][0])

    # probs = pd.DataFrame(vals)
    # probs.to_csv(path)

    # Get lowest 10% of self-matching
    probs = pd.read_csv(path, index_col=0)
    self_matches = probs[probs["Proteomic"] == probs["RNA"]]
    bad_matches = self_matches.sort_values(by=["Probability"]).head(16)['RNA'].tolist()
    for bad_match in bad_matches:
        options = probs[(probs["Proteomic"] == bad_match) | (probs["RNA"] == bad_match)]
        best_match = options.sort_values(by="Probability", ascending=False).head(1)
        rna_match = best_match['RNA'].iloc[0]
        pro_match = best_match['Proteomic'].iloc[0]
        match = rna_match if rna_match != bad_match else pro_match
        prob = best_match['Probability'].iloc[0]
        adj_prob = prob - probs[(probs["RNA"] == match) & (probs["Proteomic"] == match)]['Probability'].iloc[0]
        print(bad_match, match, adj_prob)


    # For bottom 10%, find best matches

    # Output new matches: [[1,1], [2,3], [3,2], [4,4]]
    pass
