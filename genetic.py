import pandas as pd
from random import sample, choice, randrange
from time import time

POPULATION_SIZE = 100
N_FITTEST = 10
N_GENERATIONS = 20
N_MUTATIONS = 2

pairwise_scores = pd.read_csv("data/tidy/output/pairwise_scores.csv", index_col=0)

def sum_prob(truth, individual):
    tot = 0.0
    for p1, p2 in zip(truth, individual):
        score1 = pairwise_scores[(pairwise_scores['RNA'] == p1) & (pairwise_scores['Proteomic'] == p2)]
        score2 = pairwise_scores[(pairwise_scores['RNA'] == p2) & (pairwise_scores['Proteomic'] == p1)]
        if p1 == p2:
            # Add reward for matching the samples with their original labels
            # proportional to their siamese network outputs.
            tot += float(score1["Probability"])
        tot += float(score1["Probability"]) + float(score2["Probability"])
    return tot

def crossover(seq1, seq2, crossover_index):
    offspring = seq1[:crossover_index]
    used = set(offspring)
    for item in seq2:
        if item not in used:
            offspring += [item]
        used.add(item)
    return offspring

def mutate(seq):
    seq = seq[:]
    for _ in range(N_MUTATIONS):
        loc1 = randrange(len(seq))
        loc2 = randrange(len(seq))
        seq[loc1], seq[loc2] = seq[loc2], seq[loc1]
    return seq

if __name__ == "__main__":
    # Generate initial population
    patients = pairwise_scores["RNA"].drop_duplicates().tolist()
    print("Score to beat: {}".format(sum_prob(patients, patients)))
    population = [sample(patients, len(patients)) for _ in range(POPULATION_SIZE)]

    for generation in range(N_GENERATIONS):
        # Calc scores
        scores = [sum_prob(patients, individual) for individual in population]

        # Select best individuals
        df = pd.DataFrame({"genes": population, "score": scores})
        fittest = df.sort_values(["score"]).tail(N_FITTEST)

        print("Generation {}. Best score: {}".format(generation, fittest['score'].tolist()[-1]))

        fittest = fittest['genes'].tolist()
        best_genes = fittest[-1]

        # Cross over best individuals
        halfway = int(len(patients) / 2)
        offspring = [crossover(choice(fittest), choice(fittest), halfway) for _ in range(POPULATION_SIZE)]

        # Introduce random variation
        population = [mutate(individual) for individual in offspring]

    print("Best arrangement found:")
    for p1, p2 in zip(patients, best_genes):
        print("{},{}".format(p1, p2))
