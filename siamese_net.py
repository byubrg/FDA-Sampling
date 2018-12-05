import pandas as pd

from keras.layers import Input, Dense, concatenate
from keras.models import Model

from load_data import LoadData
import feature_selection 

if __name__ == "__main__":
    data = LoadData()
    pro_data = feature_selection.univariate(data.proteomic, data.clinical)
    proteomic = Input((pro_data.shape[-1],))
    p1 = Dense(256, activation='relu')(proteomic)

    rna_data = feature_selection.univariate(data.rna, data.clinical)
    rna = Input((rna_data.shape[-1],))
    r1 = Dense(256, activation='relu')(rna)

    x = concatenate([p1, r1])
    x = Dense(64, activation='relu')(x)
    x = Dense(8, activation='relu')(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[proteomic, rna], outputs=[output])

    model.compile(optimizer='adam', loss='binary_crossentropy')

    prot_x = pd.concat([pro_data, pro_data])
    shuffled_rna = rna_data.sample(frac=1)
    rna_x = pd.concat([rna_data, shuffled_rna])
    labels = [1.0] * 80 + [0.0] * 80

    model.fit([prot_x, rna_x], [labels], epochs=20, batch_size=5)

    truth = pd.read_csv("./data/tidy/sum_tab_2.csv")
    truth['Score'] = model.predict([pro_data, rna_data])

    truth.to_csv("./data/tidy/output/siamese_scores.csv", index=False)
