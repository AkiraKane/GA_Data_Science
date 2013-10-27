
library(ggplot2)


nytimes = read.csv('/users/alexandersedgwick/dropbox/development/ga/data/sandbox/nytimes_clean.csv')


# for (i in 1:30) {
#   url_data <- read.csv(url(paste("http://stat.columbia.edu/~rachel/datasets/nyt",i,".csv", sep="")))
#   
#   if (exists("nytimes") == T) {
#     nytimes <- rbind(nytimes, url_data)
#   }
#   else {
#     nytimes <- url_data
#   }
#   rm(url_data)
# }
# 
# save(nytimes, file="/users/alexandersedgwick/dropbox/development/ga/R Projects/nytimes/nytimes.rda")
# load("/users/alexandersedgwick/dropbox/development/ga/R Projects/nytimes/nytimes.rda")
# 
# 
# names(nytimes)
# 
# nytimes$Ctr <- nytimes$Clicks/nytimes$Impressions
# 
# model <- lm(nytimes$Ctr ~ nytimes$Age + nytimes$Gender)

nytimes$Gender[nytimes$Gender==1] <- 'M'
nytimes$Gender[nytimes$Gender==0] <- 'F'

ggplot(nytimes, aes(x=Age, y=Ctr, group=Gender,colour=Gender)) + 
  geom_point()

library(rpart)
library(rpart.plot)

tree <- rpart(nytimes$Ctr ~ nytimes$Age + nytimes$Gender , data=nytimes, method='class')
print(tree)
plot(tree)
text(tree)








