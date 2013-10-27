
library(randomForest)

b2011 <- read.csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/baseball/baseball_training_2011.csv')
b2012 <- read.csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/baseball/baseball_test_2012.csv')

train_X <- subset(b2011, select=-c(salary))
train_y <- b2011['salary']



rm <- randomForest(log_salary ~ birthCountry + weight + height + bats + throws + teamID + G + G_batting + AB + R + H + X2B + X3B + HR + RBI + SB + CS + BB + SO + IBB + HBP + SH + SF, data = b2011,ntree=500, nodesize=2)

teams[order(-teams$tot_salary),] 

summary(teams$tot_salary)