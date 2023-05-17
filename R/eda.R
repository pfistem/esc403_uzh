library(ggplot2)
library(dplyr)

dat <- read.csv("~/esc403_skincancerclassifier/www/data/metadata.csv")

dat$dx <- as.factor(dat$dx)
dat$dx_type <- as.factor(dat$dx_type)
dat$sex <- as.factor(dat$sex)
dat$localization <- as.factor(dat$localization)
dat$dataset <- as.factor(dat$dataset)

str(dat)
table(dat$sex)
table(dat$dx_type)

library(forcats)
dat$dx <- fct_infreq(dat$dx)
dat$dx_type <- fct_infreq(dat$dx_type)
dat$sex <- fct_infreq(dat$sex)
dat$localization <- fct_infreq(dat$localization)
dat$dataset <- fct_infreq(dat$dataset)



library(GGally)
ggpairs(select(dat, dx, dx_type, age, sex))+theme_bw()



ggplot(data=dat, aes(x=dx, y=age, color=dx)) +
  geom_boxplot()+
  xlab("Type") +
  ylab("Age") +
  theme_bw()

ggplot(data=dat, aes(x = dx)) +
  geom_histogram(stat = "count", aes(fill=dx))+
  xlab("Type") +
  ylab("Frequency") +
  theme_bw()+
  geom_text(aes(label = ..count..), stat = "count", vjust = -0.5, colour = "black")



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
