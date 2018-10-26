#Tidyverse must be installed as a package
library(tidyverse)
setwd("/Users/Joey/byubrg/FDAsampling/FDA-Sampling/data/tidy")

#Load Data
labels = read_csv("sum_tab_1.csv")
proteins = read_csv("train_pro.csv")
msi = read_csv("train_cli.csv")

#This was useless
msiLabels = full_join(msi, labels, by = 'sample')
proteinLabels = full_join(protein, labels, by = 'sample')

msiLabelmatched = filter(msiLabels, mismatch = 0)
msiLabelmatched = transform(msiLabels, sample = sample(sample))

combinedData = full_join(msiProteins, labels, by = 'sample')
onlyMismatch = filter(combinedData, mismatch == 1)
onlyMatched = filter(combinedData, mismatch == 0)

#This first part is creating mismatches bewteen the msi and the samples
msiMatched = full_join(msi, labels, by = 'sample')
msiMismatched = full_join(transform(msiMatched, sample = sample(sample)), proteins, by = 'sample')
#Data is combined and will be corrected in the following
msiMismatched = mutate(msiMismatched, sample = paste0(sample, '0'))
msiMismatched =  msiMismatched %>% mutate(mismatch = replace_na(mismatch, 1))
msiMismatched =  msiMismatched %>% mutate(mismatch = replace(mismatch, mismatch == 0, 1))

#This will create mismatches between the proteins and the samples
proteinMatched = filter(full_join(proteins, labels, by = 'sample'), mismatch == 0)
proteinMismatched = full_join(transform(proteinMatched, sample = sample(sample)), msi, by = 'sample')
#Data is mixed and combined and needs to be corrected
proteinMismatched = mutate(proteinMismatched, sample = paste0(sample, '5'))
proteinMismatched = proteinMismatched %>% mutate(mismatch = replace_na(mismatch, 1))
proteinMismatched = proteinMismatched %>% mutate(mismatch = replace(mismatch, mismatch == 0, 1))

#add all matched and mismatched
allData = union(proteinMismatched, msiMismatched)

