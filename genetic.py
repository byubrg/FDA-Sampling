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

def cycle_crossover(seq1, seq2):
    """Cycle crossover technique for maintaining position-based genes.
    Idea came from https://stackoverflow.com/a/14423240/6481442 , and
    implementation details came from
    http://www.rubicite.com/Tutorials/GeneticAlgorithms/CrossoverOperators/CycleCrossoverOperator.aspx
    """
    ## Find the cycles
    visited_indices = set()
    d1 = {patient: index for index, patient in enumerate(seq1)}
    d2 = {patient: index for index, patient in enumerate(seq2)}
    cycles = []
    for i, (p1, p2) in enumerate(zip(seq1, seq2)):
        if i in visited_indices:
            continue
        cycle = [i]
        visited_indices.add(i)
        cycle_start = p1
        match = p2
        visited_indices.add(cycle_start)
        while match != cycle_start:
            index = d1[match]
            cycle.append(index)
            visited_indices.add(index)
            match = seq2[index]
        cycles.append(cycle)
    ## Cross them over
    child = [None] * len(seq1)
    parent = seq2
    for cycle in cycles:
        for index in cycle:
            child[index] = parent[index]
        if parent == seq1:
            parent = seq2
        else:
            parent = seq1
    return child

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
        offspring = [cycle_crossover(choice(fittest), choice(fittest)) for _ in range(POPULATION_SIZE)]

        # Introduce random variation
        population = [mutate(individual) for individual in offspring]

    print("Best arrangement found:")
    for p1, p2 in zip(patients, best_genes):
        print("{},{}".format(p1, p2))
