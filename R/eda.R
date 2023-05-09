library(ggplot2)

dat <- read.csv("~/Git/esc403_skincancerclassifier/www/data/metadata.csv")

dat$dx <- as.factor(dat$dx)
dat$dx_type <- as.factor(dat$dx_type)
dat$sex <- as.factor(dat$sex)
dat$localization <- as.factor(dat$localization)
dat$dataset <- as.factor(dat$dataset)

str(dat)
table(dat$sex)

library(dplyr)
library(GGally)
ggpairs(select(dat, dx, dx_type, age, sex))+theme_bw()








# count the number of observations per group
counts <- table(dat$dx)

# order the levels by the counts, in decreasing order
ordered_levels <- names(counts)[order(counts, decreasing = TRUE)]

# use the ordered levels to relevel the factor variable
dat$dx <- factor(dat$dx, levels = ordered_levels)

# check the new order of the levels
levels(dat$dx)


ggplot(data=dat, aes(x=dx, y=age, color=dx)) +
  geom_boxplot()+
  xlab("Grazing treatment") +
  ylab("Fruit Production") +
  theme_bw()

ggplot(data=dat, aes(x = dx)) +
  geom_histogram(stat = "count", aes(fill=dx))+
  theme_bw()

