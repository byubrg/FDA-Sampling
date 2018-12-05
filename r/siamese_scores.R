if (!require("pacman")) install.packages("pacman"); library(pacman)
p_load("tidyverse")
read_csv("data/tidy/output/siamese_scores.csv") %>%
    mutate(matching = RNAseq == Proteomics) %>%
    group_by(matching) %>%
    summarize(avg_score = mean(Score))
