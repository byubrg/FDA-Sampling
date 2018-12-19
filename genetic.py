import pandas as pd
from random import sample, choice, randrange
from probabilities import clinical_labels_dict

POPULATION_SIZE = 1000
N_FITTEST = 100
N_GENERATIONS = 500
N_MUTATIONS = 1
TEST_OR_TRAIN = "train"
MATCHING_BONUS = 1.1

class Individual(object):
    def __init__(self, clinical=None, proteomic=None, rna=None):
        self.clinical = clinical
        self.proteomic = proteomic
        self.rna = rna

    def copy(self):
        return Individual(self.clinical[:], self.proteomic[:], self.rna[:])

    def dataframe(self):
        df = pd.DataFrame({
            "Clinical": self.clinical,
            "RNAseq": self.rna,
            "Proteomics": self.proteomic
        }, index=self.clinical)

        df.index = df.index.rename("sample")
        df = df[['Clinical', 'RNAseq', 'Proteomics']]
        order = [df.index[0].split("_")[0] + "_{}".format(i) for i in range(1, len(df.index)+1)]
        df = df.reindex(order)
        df = df.applymap(lambda cell: cell.split("_")[-1])

        return df

    def mutate(self):
        return Individual(mutate(self.clinical), mutate(self.proteomic), mutate(self.rna))

class Genetic(object):
    def __init__(self, train=True):
        if train:
            rna_prot_df = pd.read_csv("./data/probabilities/rna_proteomic.csv", index_col=0)
            rna_clin_df = pd.read_csv("./data/probabilities/clinical_rna.csv", index_col=0)
            prot_clin_df = pd.read_csv("./data/probabilities/clinical_proteomic.csv", index_col=0)
        else:
            rna_prot_df = pd.read_csv("./data/probabilities/rna_proteomic_test.csv", index_col=0)
            rna_clin_df = pd.read_csv("./data/probabilities/clinical_rna_test.csv", index_col=0)
            prot_clin_df = pd.read_csv("./data/probabilities/clinical_proteomic_test.csv", index_col=0)

        self.rna_prot = {}
        for i, row in rna_prot_df.iterrows():
            for prot, prob in row.iteritems():
                rna = row.name
                if rna not in self.rna_prot:
                    self.rna_prot[rna] = {}
                self.rna_prot[rna][prot] = prob
        self.rna_clin = {}
        for i, row in rna_clin_df.iterrows():
            self.rna_clin[row.name] = row.tolist()
        self.prot_clin = {}
        for i, row in prot_clin_df.iterrows():
            self.prot_clin[row.name] = row.tolist()
        self.patients = prot_clin_df.index.tolist()
        self.clin = clinical_labels_dict(train=train)
        self.initialize_population()

    def mutate(self, individual):
        return individual.mutate()

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        clinical = cycle_crossover(parent1.clinical, parent2.clinical)
        proteomic = cycle_crossover(parent1.proteomic, parent2.proteomic)
        rna = cycle_crossover(parent1.rna, parent2.rna)
        return Individual(clinical=clinical, proteomic=proteomic, rna=rna)

    def fitness(self, individual, matching_bonus=MATCHING_BONUS):
        score = 0.0
        for clin, prot, rna in zip(individual.clinical, individual.proteomic, individual.rna):
            prot_clin_label = int(self.clin[prot])
            rna_clin_label = int(self.clin[rna])
            if clin != prot and clin != rna and prot != rna:
                score -= 1
            rna_prot_score = self.rna_prot[rna][prot]
            rna_prot_score *= matching_bonus if rna == prot else 1.0
            rna_clin_score = float(self.rna_clin[clin][rna_clin_label])
            rna_clin_score *= matching_bonus if rna == clin else 1.0
            prot_clin_score = float(self.prot_clin[clin][prot_clin_label])
            prot_clin_score *= matching_bonus if prot == clin else 1.0
            score += rna_prot_score + rna_clin_score + prot_clin_score
        return score

    def initialize_population(self):
        self.population = [
            Individual(*[
                sample(self.patients, len(self.patients)) for _ in range(3)
            ]) for _ in range(POPULATION_SIZE)
        ]

    def generation(self):
        # Calc scores
        scores = [self.fitness(individual) for individual in self.population]
        # Select best individuals
        df = pd.DataFrame({"genes": self.population, "score": scores})
        fittest_df = df.sort_values(["score"]).tail(N_FITTEST)
        fittest = fittest_df['genes'].tolist()
         # Give the best individual we've seen extra reproductive chances.
        if fittest_df['score'].tolist()[-1] > self.best_score:
            self.best_score = fittest_df['score'].tolist()[-1]
            self.best_genes = fittest[-1]
        fittest.append(self.best_genes)
        # Cross over best individuals
        offspring = [
            self.crossover(choice(fittest), choice(fittest)) for _ in range(POPULATION_SIZE)
        ]
        # Introduce random variation
        self.population = [self.mutate(individual) for individual in offspring]

    def train(self):
        self.best_genes = Individual(
            mutate(self.patients, 2), mutate(self.patients, 2), mutate(self.patients, 2)
        )
        self.best_score = self.fitness(self.best_genes)
        for generation in range(N_GENERATIONS):
                # Train without mutations for the last 5% of generations.
            if generation > N_GENERATIONS * 0.95:
                N_MUTATIONS = 0
            self.generation()
            print("Generation {}. Best score: {}".format(generation, self.best_score))

def cycle_crossover(parent1, parent2):
    """Cycle crossover technique for maintaining position-based genes.
    Idea came from https://stackoverflow.com/a/14423240/6481442 , and
    implementation details came from
    http://www.rubicite.com/Tutorials/GeneticAlgorithms/CrossoverOperators/CycleCrossoverOperator.aspx
    """
    ## Find the cycles
    visited_indices = set()
    d1 = {patient: index for index, patient in enumerate(parent1)}
    d2 = {patient: index for index, patient in enumerate(parent2)}
    cycles = []
    for i, (p1, p2) in enumerate(zip(parent1, parent2)):
        if i in visited_indices:
            continue
        cycle = [i]
        visited_indices.add(i)
        cycle_start = p1
        match = p2
        while match != cycle_start:
            index = d1[match]
            cycle.append(index)
            visited_indices.add(index)
            match = parent2[index]
        cycles.append(cycle)
    ## Cross them over
    child = [None] * len(parent1)
    parent = parent2
    for cycle in cycles:
        for index in cycle:
            child[index] = parent[index]
        if parent == parent1:
            parent = parent2
        else:
            parent = parent1
    return child

def mutate(seq, n_mutations=N_MUTATIONS):
    seq = seq[:]
    for _ in range(n_mutations):
        loc1 = randrange(len(seq))
        loc2 = randrange(len(seq))
        seq[loc1], seq[loc2] = seq[loc2], seq[loc1]
    return seq

def most_common(lst):
    return max(set(lst), key=lst.count)

def reorder_sample_ids(df):
    print('Hello Mundo')
    #take df loop over each row
    for i in range(0,len(df.index)):
        most_common_index = int(most_common([df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2]]))
        current_index = i + 1
        if most_common_index is not current_index:
            # switch content of the two indices
            # using -1 because samples ids are 1 based but python is 0 based
            temp_values = df.iloc[(most_common_index-1),:].copy()
            df.iloc[(most_common_index-1),:] = df.iloc[i,:].copy()
            df.iloc[i, :] = temp_values
    return df


if __name__ == "__main__":
    genetic = Genetic(train=True)
    no_mismatches = Individual(genetic.patients, genetic.patients, genetic.patients)
    print("Scores for training set. Score to beat: {}".format(genetic.fitness(no_mismatches)))

    genetic.train()

    print("Best arrangement found:")
    df = genetic.best_genes.dataframe()
    print(df)
    df.to_csv("./data/tidy/output/siamese_submission_train.csv")

    print()

    genetic_test = Genetic(train=False)
    no_mismatches = Individual(genetic_test.patients, genetic_test.patients, genetic_test.patients)
    print("Scores for test set. Score to beat: {}".format(genetic_test.fitness(no_mismatches)))

    genetic_test.train()

    print("Best arrangement found:")
    df = genetic_test.best_genes.dataframe()
    print(df)
    reorder_sample_ids(df)
    print(df)
    df.to_csv("./data/tidy/output/siamese_submission_test.csv")
