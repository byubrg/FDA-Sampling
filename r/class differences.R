library(tidyverse)
library(plyr)


joined.data.frame <- as.data.frame(joined.data)
joined.data.frame[is.na(joined.data.frame)] <- 0
bin <- mapvalues(joined.data.frame$msi, from = c("MSI-High", "MSI-Low/MSS"), to = c(1,0))
bin <- as.data.frame(as.factor(bin))
joined.data.frame <- data.frame(cbind(joined.data.frame, bin))

male.high <- as.data.frame(filter(joined.data, gender == 'Male', msi == "MSI-High"))
female.high <- as.data.frame(filter(joined.data, gender == 'Female', msi == "MSI-High"))
male.low <- as.data.frame(filter(joined.data, gender == 'Male', msi == "MSI-Low/MSS"))
female.low <- as.data.frame(filter(joined.data, gender == 'Female', msi == "MSI-Low/MSS"))


means1<- as.data.frame(lapply(male.high[,4:4121], FUN = mean, na.rm = T))
means1 <- cbind(genes, t(means1))
colnames(means1) <- c('gene', 'value')

means2<- as.data.frame(lapply(female.high[,4:4121], FUN = mean, na.rm = T))
means2 <- cbind(genes, t(means2))
colnames(means2) <- c('gene', 'value')


means3<- as.data.frame(lapply(male.low[,4:4121], FUN = mean, na.rm = T))
means3 <- cbind(genes, t(means3))
colnames(means3) <- c('gene', 'value')

means4<- as.data.frame(lapply(female.low[,4:4121], FUN = mean, na.rm = T))
means4 <- cbind(genes, t(means4))
colnames(means4) <- c('gene', 'value')



par(mfrow=c(2,2))
plot_grid(mh, fh, ml, fl, ncol = 2, nrow = 2)

mh <- ggplot(data = means1, mapping = aes(means1$gene, means1$value))+
  geom_point(color = "blue", alpha = .2)+
  theme(axis.text.x = element_blank())
fh <- ggplot(data = means2, mapping = aes(means2$gene, means2$value))+
  geom_point(color = "red", alpha = .2)+
  theme(axis.text.x = element_blank())
ml <- ggplot(data = means3, mapping = aes(means3$gene, means3$value))+
  geom_point(color = "blue", alpha = .2)+
  theme(axis.text.x = element_blank())
fl <- ggplot(data = means4, mapping = aes(means4$gene, means4$value))+
  geom_point(color = "red", alpha = .2)+
  theme(axis.text.x = element_blank())

diffs <- as.numeric(means1$value-means3$value)
sort(diffs, decreasing = T)
d1 <- as.data.frame(diffs)

diffs2 <- as.numeric(means2$value - means4$value)
sort(diffs2, decreasing = T)
d2 <- as.data.frame(diffs2)

d3 <- cbind(d1,d2)
diffmales <- cbind(genes, diffs)
difffemales <-cbind(genes, diffs2)

ggplot(data = diffmales, mapping = aes(diffmales$`ls(means, all.names = T)`, diffmales$diffs))+
  geom_point(color = "blue", alpha = .2)+
  theme(axis.text.x = element_blank())

ggplot(data = difffemales, mapping = aes(difffemales$`ls(means, all.names = T)`, difffemales$diffs2))+
  geom_point(color = "red", alpha = .2)+
  theme(axis.text.x = element_blank())
 
str(as.data.frame(joined.data[, 4:4121]))
str(as.data.frame(joined.data[,3]))
#feature selection 
# ensure the results are repeatable
set.seed(7)
# load the library
library(mlbench)
library(caret)
#load the data
data(joined.data.frame)
#define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(joined.data.frame[,4:4122], joined.data.frame[,4122], sizes=c(1:8), rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))

data("PimaIndiansDiabetes")
View(PimaIndiansDiabetes)
