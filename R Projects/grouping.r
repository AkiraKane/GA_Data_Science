
library(randomForest)

l_train = read.csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/lemons/lemon_training.csv')
l_test =read.csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/lemons/lemon_test.csv')


data <-  l_train[, !(colnames(l_train) %in% c("IsBadBuy"))]
target <- l_train['IsBadBuy']  

library(gpairs)
data <- l_train[c('IsBadBuy','VehYear','VehicleAge','Transmission','VehOdo','WheelType','Size','Nationality','TopThreeAmericanName','IsOnlineSale','WarrantyCost')]

pairs(data)


#Random Forest Model
library(dummies)
l_train <- l_train[complete.cases(l_train),]


l_train <- cbind(l_train, dummy('VNST',data=l_train))
#dummy(l_train$Model)
l_train <- cbind(l_train, dummy('Auction',data=l_train))
#l_train <- cbind(l_train, dummy('Make',data=l_train))
#dummy(l_train$Trim)
#dummy(l_train$SubModel)
#dummy(l_train$Color)
l_train <- cbind(l_train, dummy('Transmission',data=l_train))
l_train <- cbind(l_train, dummy('WheelType',data=l_train))
l_train <- cbind(l_train, dummy('Nationality',data=l_train))
l_train <- cbind(l_train, dummy('Size',data=l_train))
l_train <- cbind(l_train, dummy('TopThreeAmericanName',data=l_train))
l_train <- cbind(l_train, dummy('PRIMEUNIT',data=l_train))
l_train <- cbind(l_train, dummy('AUCGUART',data=l_train))



l_train$VNST <- NULL
l_train$Model <- NULL
l_train$Auction <- NULL
l_train$Make <- NULL
l_train$Trim <- NULL
l_train$SubModel <- NULL
l_train$Color <- NULL
l_train$Transmission <- NULL
l_train$WheelType <- NULL
l_train$Nationality <- NULL
l_train$Size <- NULL
l_train$TopThreeAmericanName <- NULL
l_train$PRIMEUNIT <- NULL
l_train$AUCGUART <- NULL
l_train$VNST <- NULL
l_train$RefId <- NULL
l_train$PurchDate <- NULL

names(l_train) <- sub(" ", ".", names(l_train))
names(l_train) <- sub(" ", ".", names(l_train))


rm <- randomForest(as.factor(l_train$IsBadBuy) ~ VehicleAge+VehOdo+VNZIP1+IsOnlineSale+TransmissionManual, data = l_train,ntree=500, nodesize=2)

rm$confusion
rm$votes
rm$err.rate




library(ggplot2)
plotmatrix(with(l_train, data.frame(Auction, VehicleAge, Transmission, VehOdo)))

#GBM



