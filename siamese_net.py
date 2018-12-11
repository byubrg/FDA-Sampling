import pandas as pd

from keras.layers import Input, Dense, concatenate
from keras.models import Model

from load_data import LoadData
import feature_selection

class SiameseNet(object):
    def __init__(self, x_shapes):
        # input layers
        proteomic = Input(x_shapes[0])
        rna = Input(x_shapes[1])
        # dense layers
        p1 = Dense(256, activation='relu')(proteomic)
        r1 = Dense(256, activation='relu')(rna)
        # concatenate the 2 input branches
        x = concatenate([p1, r1])
        x = Dense(64, activation='relu')(x)
        x = Dense(8, activation='relu')(x)
        # output stuff
        output = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs=[proteomic, rna], outputs=[output])
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

    def fit(self, x, y, **kwargs):
        """Train the siamese network. `x` should be a pair of 2D arrays
        with the same length:
        [
            [[0, 1, 0], [1, 1, 0]], # i.e. RNA-seq,   2 samples, 3 genes
            [[1, 0],    [1, 1]   ], # i.e. proteomic, 2 samples, 2 genes
        ]
        `y` is mismatch labels:
            if x[0][i] matches x[1][i], y[i] == 1.0, else 0.0
        """
        self.model.fit(x, [y], **kwargs)

    def predict(self, x):
        """`x` is the same as in the `fit()` method.
        Returns a list `y` of probabilities of x[0][i] matching x[1][i]
        """
        return self.model.predict(x)


if __name__ == "__main__":
    data = LoadData()
    pro_data = feature_selection.univariate(data.proteomic, data.clinical)
    rna_data = feature_selection.univariate(data.rna, data.clinical)

    prot_x = pd.concat([pro_data, pro_data])
    shuffled_rna = rna_data.sample(frac=1)
    rna_x = pd.concat([rna_data, shuffled_rna])
    labels = [1.0] * 80 + [0.0] * 80

    network = SiameseNet([
        (pro_data.shape[-1],),
        (rna_data.shape[-1],)
    ])
    network.fit([prot_x, rna_x], labels, epochs=500, batch_size=5)

    truth = pd.read_csv("./data/tidy/sum_tab_2.csv")
    truth['Score'] = network.predict([pro_data, rna_data])

    truth.to_csv("./data/tidy/output/siamese_scores.csv", index=False)
