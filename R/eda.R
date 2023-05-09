library(ggplot2)

dat <- read.csv("~/Git/esc403_skincancerclassifier/www/data/metadata.csv")

dat$dx <- as.factor(dat$dx)
dat$dx_type <- as.factor(dat$dx_type)
dat$sex <- as.factor(dat$sex)
dat$localization <- as.factor(dat$localization)
dat$dataset <- as.factor(dat$dataset)

str(dat)
table(dat$sex)
table(dat$dx_type)


library(dplyr)
library(GGally)
ggpairs(select(dat, dx, dx_type, age, sex))+theme_bw()




# use the ordered levels to relevel the factor variable
counts <- table(dat$dx)
ordered_levels <- names(counts)[order(counts, decreasing = TRUE)]
dat$dx <- factor(dat$dx, levels = ordered_levels)

ggplot(data=dat, aes(x=dx, y=age, color=dx)) +
  geom_boxplot()+
  xlab("") +
  ylab("") +
  theme_bw()

ggplot(data=dat, aes(x = dx)) +
  geom_histogram(stat = "count", aes(fill=dx))+
  theme_bw()



library(tableone)
table_one <- CreateTableOne(data = dat, vars=c("dx", "dx_type"),
                            addOverall = TRUE, test = FALSE)
table_one <- print(table_one, missing=TRUE)

kableone(table_one, booktabs = T, format = "latex")



table_one <- CreateTableOne(data = dat, vars=c("age", "sex","localization"),
                            addOverall = TRUE, test = FALSE)
table_one <- print(table_one, missing=TRUE)

kableone(table_one, booktabs = T, format = "latex")
