# 
# playerID       Player ID code
# yearID         Year
# stint          player's stint (order of appearances within a season)
# teamID         Team
# lgID           League
# G              Games
# G_batting      Game as batter
# AB             At Bats
# R              Runs - Team
# H              Hits
# 2B             Doubles
# 3B             Triples
# HR             Homeruns - Ind
# RBI            Runs Batted In
# SB             Stolen Bases
# CS             Caught Stealing
# BB             Base on Balls
# SO             Strikeouts
# IBB            Intentional walks
# HBP            Hit by pitch
# SH             Sacrifice hits
# SF             Sacrifice flies
# GIDP           Grounded into double plays
# G_Old          Old version of games (deprecated)

b2011 <- read.csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/baseball/baseball_training_2011.csv')
b2012 <- read.csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/baseball/baseball_test_2012.csv')

train_X <- subset(b2011, select=-c(salary))
train_y <- b2011['salary']

#test_X = b2012[c('G', 'AB', 'R', 'H', 'X2B', 'X3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF')]
test_X <- subset(b2012, select=-c(salary))
b2012_csv = b2012[c('playerID','yearID','salary')]

#Inital Model

fit = lm(salary ~ G+ AB + R + H + X2B + X3B + HR + RBI + SB + CS + BB + SO + IBB + HBP + SH + SF,b2011)
summary(fit)
anova(fit)
anova(fit)[["Mean Sq"]] 

MSE = (summary(fit)$sigma)^2


# diagnostic plots 
layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page 
plot(fit)


# Stepwise Regression
library(MASS)
step <- stepAIC(fit, direction="both")
step$anova # display results

str(step)

# All Subsets Regression
library(leaps)
#ttach(mydata)
leaps<-regsubsets(salary ~ G+ AB + R + H + X2B + X3B + HR + RBI + SB + CS + BB + SO + IBB + HBP + SH + SF,data=b2011,nbest=10)
# view results 
summary(leaps)
# plot a table of models showing variables in each model.
# models are ordered by the selection statistic.
plot(leaps,scale="r2")
plot(leaps)
# plot statistic by subset size 
library(car)
subsets(leaps, statistic="rsq")




salary ~ G + AB + R + H + X2B + X3B + RBI + CS + SO + IBB


fit = lm(salary ~ G+ AB + R + H + X2B + X3B + HR + RBI + SB + CS + BB + SO + IBB + HBP + SH + SF,b2011)
fit2 = lm(salary ~ G + AB + R + H + X2B + X3B + RBI + CS + SO + IBB,b2011)
summary(fit)
summary(fit2)

anova(fit)
anova(fit)[["Mean Sq"]] 




# Calculate Relative Importance for Each Predictor
library(relaimpo)
calc.relimp(fit,type=c("lmg","last","first"),
            rela=TRUE)

# Bootstrap Measures of Relative Importance (1000 samples) 
boot <- boot.relimp(fit, b = 1000, type = c("lmg", 
                                            "last", "first"), rank = TRUE, 
                    diff = TRUE, rela = TRUE)
booteval.relimp(boot) # print result
plot(booteval.relimp(boot,sort=TRUE)) # plot result


####
# Look at all variables
####

b2011$log_salary <- log(b2011$salary)

teams <- ddply(b2011,.(b2011$teamID), summarize, tot_salary=sum(salary))


b2011$team_group <- 


fit <- lm(log_salary ~ birthCountry + weight + height + bats + throws + teamID + G + G_batting + AB + R + H + X2B + X3B + HR + RBI + SB + CS + BB + SO + IBB + HBP + SH + SF ,b2011)

library(MASS)
step <- stepAIC(fit, direction="both")
step$anova # display results



# Optimized Best Fit

fit <- lm(salary ~ weight + height  + G + AB + R + H + X3B + RBI + CS + +BB + SO,b2011)
summary(fit)
anova(fit)
anova(fit)[["Mean Sq"]] 


b2012_csv$predicted <- predict(fit,test_X)





# Pricipal Components Analysis
# entering raw data and extracting PCs 
# from the correlation matrix 
princom <- princomp(b2011[c('weight', 'height'  , 'G' , 'AB' , 'R' , 'H' , 'X2B' , 'X3B' , 'HR' , 'RBI' , 'CS' , 'SO' , 'IBB')], cor=TRUE)
summary(princom) # print variance accounted for 
loadings(princom) # pc loadings 
plot(princom,type="lines") # scree plot 
princom$scores # the principal components
biplot(princom)

plot(princom)
barplot(princom$sdev/princom$sdev[1])

princom <- princomp(b2011[c('weight', 'height'  , 'G' , 'AB' , 'R' , 'H' , 'X2B' , 'X3B' , 'HR' , 'RBI' , 'CS' , 'SO' , 'IBB')], tol=.1,cor=TRUE)
summary(princom)

plot(princom$x)


# Determine Number of Factors to Extract
library(nFactors)
ev <- eigen(cor(mydata)) # get eigenvalues
ap <- parallel(subject=nrow(mydata),var=ncol(mydata),
               rep=100,cent=.05)
nS <- nScree(x=ev$values, aparallel=ap$eigen$qevpea)
plotnScree(nS)