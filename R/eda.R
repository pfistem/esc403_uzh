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


dat <- transform(dat, var = factor(dx, names(sort(-table(dx)))))
dat <- transform(dat, var = factor(dx_type, names(sort(-table(dx_type)))))
dat <- transform(dat, var = factor(sex, names(sort(-table(sex)))))
dat <- transform(dat, var = factor(localization, names(sort(-table(localization)))))
dat <- transform(dat, var = factor(dataset, names(sort(-table(dataset)))))


library(dplyr)
library(GGally)
ggpairs(select(dat, dx, dx_type, age, sex))+theme_bw()



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


library(naniar)
gg_miss_var(dat,  show_pct = TRUE)
