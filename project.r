setwd("D:\\Programming\\John Hopkins\\John_Hopkins_Data_Science\\8. Practical Machince Learning\\project\\")

training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")

head(training)

#####Select features related to acceleration
accel <- grepl("accel|classe", names(training))

dfTraining <- training[, accel]
head(dfTraining)

#Remove columns with NA
NACols <- sapply(dfTraining, function(x) {
  sum(is.na(x))
})

selectedCols <- NACols == 0

####Clean training dataset 
dataTrain <- dfTraining[, selectedCols]
head(dataTrain)


###################Testing Data###########
accel <- grepl("accel", names(testing))

dfTesting <- testing[, accel]

####Remove columns with NA
NACols <- sapply(dfTesting, function(x) {
  sum(is.na(x))
})

selectedCols <- NACols == 0

###Cleaned test dataset
dataTest <- dfTesting[, selectedCols]
head(dataTest)


dim(training)
dim(dataTrain)

dim(testing)
dim(dataTest)



#######################################################################
library(ggplot2)
library(gridExtra)
library(corrplot)
library(rpart)
library(rattle)
library(rpart.plot)
library(caret)

g1 <- ggplot(dataTrain, aes(total_accel_belt, total_accel_dumbbell, col = classe)) +
  geom_point(size = 4, alpha = 0.2)

g2 <- ggplot(dataTrain, aes(total_accel_dumbbell, total_accel_forearm, col = classe)) +
  geom_point(size = 4, alpha = 0.2)

grid.arrange(g1, g2, ncol = 2)

corrplot(cor(dataTrain[, -length(names(dataTrain))]), method = "color", tl.cex = 0.5)

##############Decision Tree
modelTree <- rpart(classe ~ ., data = dataTrain, method = "class")
prp(modelTree)

#####Random Forest###########
modelRF <- train(classe ~ ., data = dataTrain, method = "rf", trControl = trainControl(method = "cv", 5), ntree = 250)
modelRF

predict(modelRF, dataTest)



#################SVM##########
library(e1071)

dataTrain2 <- dataTrain

dataTrain2$classe <- as.factor(dataTrain2$classe)
head(dataTrain2)
modelSvm <- tune(svm, classe ~ .,
                 data = dataTrain2,
                 kernel = "radial",
                 ranges = list(cost = c(0.01, 0.1, 1, 10, 100),
                               gamma = c(0.5, 1, 2)))

set.seed(4621)
svmfit <- svm(classe ~ ., data = dataTrain2,
              kernel="radial", cost = 10, gamma=1,
              scale = TRUE,
              cross = 5)

svmfit

predictions <- c()

pred <- predict(svmfit, dataTest, decision.values = TRUE)
pred

for (i in 1:length(pred)) {
  if (pred[i] < 2) {
    predictions[i] <- 'A'
  } else if (pred[i] < 3 & pred[i] > 2) {
    predictions[i] <- 'B'
  } else if(pred[i] < 4 & pred[i] > 3) {
    predictions[i] <- 'C'
  } else if (pred[i] < 5 & pred[i] > 4) {
    predictions[i] <- 'D'
  } else {
    predictions[i] <- 'E'
  }
}

as.factor(predictions)

svmfit$tot.MSE


