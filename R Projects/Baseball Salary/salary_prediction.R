



baseball <- read.csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/baseball/baseball.csv')

baseball_mod <- baseball[c('HR', 'RBI', 'R', 'G', 'SB', 'salary', 'height', 'weight', 'yearID')]


head(baseball_mod)

attach(baseball_mod)

baseball_mod$Age <- baseball_mod$yearID - baseball_mod$birthYear
baseball_mod <- baseball_mod[complete.cases(baseball_mod),]

model_1 <- lm(salary~HR + RBI + R + G + SB + height + weight + yearID)

cov(baseball_mod)
cor(baseball_mod)


#Using Leaps - http://www.statmethods.net/stats/regression.html

# All Subsets Regression
library(leaps)
attach(baseball_mod)
leaps<-regsubsets(salary~HR + RBI + R + G + SB + height + weight + yearID,data=baseball_mod,nbest=10)
# view results 
summary(leaps)
# plot a table of models showing variables in each model.
# models are ordered by the selection statistic.
plot(leaps,scale="r2")
# plot statistic by subset size 
library(car)
subsets(leaps, statistic="rsq")


# Calculate Relative Importance for Each Predictor
library(relaimpo)
calc.relimp(model_1,type=c("lmg","last","first","pratt"),
            rela=TRUE)

# Bootstrap Measures of Relative Importance (1000 samples) 
boot <- boot.relimp(model_1, b = 1000, type = c("lmg", 
                                            "last", "first", "pratt"), rank = TRUE, 
                    diff = TRUE, rela = TRUE)
booteval.relimp(boot) # print result
plot(booteval.relimp(boot,sort=TRUE)) # plot result

# Model 2


baseball_mod <- baseball[c('HR', 'RBI', 'R', 'G', 'SB', 'salary', 'height', 'weight', 'yearID')]

baseball_mod <- baseball[c('HR', 'RBI', 'R', 'G', 'SB', 'salary', 'height', 'weight', 'yearID')]


