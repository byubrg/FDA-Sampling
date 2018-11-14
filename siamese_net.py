import pandas as pd

from keras.layers import Input, Dense, concatenate
from keras.models import Model

from load_data import LoadData

if __name__ == "__main__":
    data = LoadData()

    proteomic = Input((data.proteomic.shape[-1],))
    p1 = Dense(256, activation='relu')(proteomic)

    rna = Input((data.rna.shape[-1],))
    r1 = Dense(256, activation='relu')(rna)

    x = concatenate([p1, r1])
    x = Dense(64, activation='relu')(x)
    x = Dense(8, activation='relu')(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[proteomic, rna], outputs=[output])

    model.compile(optimizer='adam', loss='binary_crossentropy')

    prot_x = pd.concat([data.proteomic, data.proteomic])
    shuffled_rna = data.rna.sample(frac=1)
    rna_x = pd.concat([data.rna, shuffled_rna])
    labels = [1.0] * 80 + [0.0] * 80

    model.fit([prot_x, rna_x], [labels], epochs=50, batch_size=2)

    truth = pd.read_csv("./data/tidy/sum_tab_2.csv")
    truth['Score'] = model.predict([data.proteomic, data.rna])

    truth.to_csv("./data/tidy/output/siamese_scores.csv", index=False)
