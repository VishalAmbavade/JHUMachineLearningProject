---
title: "Excercise Manner Predictions"
author: "Vishal"
date: "30/06/2020"
output: 
  html_document:
    keep_md: TRUE
---



## Synopsis

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

In this project, our goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

In the following analysis I have used SVM, Random Forest and decison tree. 

## Load and preprocess data


```r
library(ggplot2)
library(gridExtra)
library(corrplot)
library(rpart)
library(rattle)
library(rpart.plot)
library(caret)
library(e1071)

training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")
```

## Data Cleaning

In order to make data suitable for predictions and discard unnecessary features from the data we will clean it. For cleaning we will consider only the columns which are related to acceleration. Then we will discard the columns containing NA values.


```r
#####Select features related to acceleration
accel <- grepl("accel|classe", names(training))

dfTraining <- training[, accel]
head(dfTraining)
```

```
##   total_accel_belt var_total_accel_belt accel_belt_x accel_belt_y accel_belt_z
## 1                3                   NA          -21            4           22
## 2                3                   NA          -22            4           22
## 3                3                   NA          -20            5           23
## 4                3                   NA          -22            3           21
## 5                3                   NA          -21            2           24
## 6                3                   NA          -21            4           21
##   total_accel_arm var_accel_arm accel_arm_x accel_arm_y accel_arm_z
## 1              34            NA        -288         109        -123
## 2              34            NA        -290         110        -125
## 3              34            NA        -289         110        -126
## 4              34            NA        -289         111        -123
## 5              34            NA        -289         111        -123
## 6              34            NA        -289         111        -122
##   total_accel_dumbbell var_accel_dumbbell accel_dumbbell_x accel_dumbbell_y
## 1                   37                 NA             -234               47
## 2                   37                 NA             -233               47
## 3                   37                 NA             -232               46
## 4                   37                 NA             -232               48
## 5                   37                 NA             -233               48
## 6                   37                 NA             -234               48
##   accel_dumbbell_z total_accel_forearm var_accel_forearm accel_forearm_x
## 1             -271                  36                NA             192
## 2             -269                  36                NA             192
## 3             -270                  36                NA             196
## 4             -269                  36                NA             189
## 5             -270                  36                NA             189
## 6             -269                  36                NA             193
##   accel_forearm_y accel_forearm_z classe
## 1             203            -215      A
## 2             203            -216      A
## 3             204            -213      A
## 4             206            -214      A
## 5             206            -214      A
## 6             203            -215      A
```

```r
#Remove columns with NA
NACols <- sapply(dfTraining, function(x) {
  sum(is.na(x))
})

selectedCols <- NACols == 0

####Clean training dataset 
dataTrain <- dfTraining[, selectedCols]
head(dataTrain)
```

```
##   total_accel_belt accel_belt_x accel_belt_y accel_belt_z total_accel_arm
## 1                3          -21            4           22              34
## 2                3          -22            4           22              34
## 3                3          -20            5           23              34
## 4                3          -22            3           21              34
## 5                3          -21            2           24              34
## 6                3          -21            4           21              34
##   accel_arm_x accel_arm_y accel_arm_z total_accel_dumbbell accel_dumbbell_x
## 1        -288         109        -123                   37             -234
## 2        -290         110        -125                   37             -233
## 3        -289         110        -126                   37             -232
## 4        -289         111        -123                   37             -232
## 5        -289         111        -123                   37             -233
## 6        -289         111        -122                   37             -234
##   accel_dumbbell_y accel_dumbbell_z total_accel_forearm accel_forearm_x
## 1               47             -271                  36             192
## 2               47             -269                  36             192
## 3               46             -270                  36             196
## 4               48             -269                  36             189
## 5               48             -270                  36             189
## 6               48             -269                  36             193
##   accel_forearm_y accel_forearm_z classe
## 1             203            -215      A
## 2             203            -216      A
## 3             204            -213      A
## 4             206            -214      A
## 5             206            -214      A
## 6             203            -215      A
```

```r
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
```

```
##   total_accel_belt accel_belt_x accel_belt_y accel_belt_z total_accel_arm
## 1               20          -38           69         -179              10
## 2                4          -13           11           39              38
## 3                5            1           -1           49              44
## 4               17           46           45         -156              25
## 5                3           -8            4           27              29
## 6                4          -11          -16           38              14
##   accel_arm_x accel_arm_y accel_arm_z total_accel_dumbbell accel_dumbbell_x
## 1          16          38          93                    9               21
## 2        -290         215         -90                   31             -153
## 3        -341         245         -87                   29             -141
## 4        -238         -57           6                   18              -51
## 5        -197         200         -30                    4              -18
## 6         -26         130         -19                   29             -138
##   accel_dumbbell_y accel_dumbbell_z total_accel_forearm accel_forearm_x
## 1              -15               81                  33            -110
## 2              155             -205                  39             212
## 3              155             -196                  34             154
## 4               72             -148                  43             -92
## 5              -30               -5                  24             131
## 6              166             -186                  43             230
##   accel_forearm_y accel_forearm_z
## 1             267            -149
## 2             297            -118
## 3             271            -129
## 4             406             -39
## 5             -93             172
## 6             322            -144
```

```r
dim(training)
```

```
## [1] 19622   160
```

```r
dim(dataTrain)
```

```
## [1] 19622    17
```

```r
dim(testing)
```

```
## [1]  20 160
```

```r
dim(dataTest)
```

```
## [1] 20 16
```

## Exploratory Data Anyalsis

1. We'll plot the some of the features and see if the data is linearly separable or not, which can further help is in understanding which algorithm to use.
2. We'll find out the correlation between the features.


```r
g1 <- ggplot(dataTrain, aes(total_accel_belt, total_accel_dumbbell, col = classe)) +
  geom_point(size = 4, alpha = 0.2)

g2 <- ggplot(dataTrain, aes(total_accel_dumbbell, total_accel_forearm, col = classe)) +
  geom_point(size = 4, alpha = 0.2)

grid.arrange(g1, g2, ncol = 2)
```

![](project_files/figure-html/unnamed-chunk-3-1.png)<!-- -->

```r
corrplot(cor(dataTrain[, -length(names(dataTrain))]), method = "color", tl.cex = 0.5)
```

![](project_files/figure-html/unnamed-chunk-3-2.png)<!-- -->

## Decision Tree
We'll see how the decision tree for our data looks like.


```r
##############Decision Tree
modelTree <- rpart(classe ~ ., data = dataTrain, method = "class")
prp(modelTree)
```

![](project_files/figure-html/unnamed-chunk-4-1.png)<!-- -->

## Random Forest
I've used Random Forest because it automatically selects important variables from the data is robust algo. 

I've used <b>5 fold cross validation</b> in the algorithm.


```r
modelRF <- train(classe ~ ., data = dataTrain, method = "rf", trControl = trainControl(method = "cv", 5), ntree = 250)
modelRF
```

```
## Random Forest 
## 
## 19622 samples
##    16 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 15699, 15698, 15696, 15698, 15697 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9509733  0.9379645
##    9    0.9466926  0.9325419
##   16    0.9331367  0.9153878
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 2.
```

```r
predict(modelRF, dataTest)
```

```
##  [1] B A C A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

This model has produced the accuracy of <b>95.12%</b> which is highest compared to any other algorithm.

## SVM

I've tried predicting the data with SVM model as well. The Non-linear SVM is used in this particular problem as data is not linearly separable.



I have got .200 MSE for the SVM Model. I have used <b>tune function</b> from <b> e1071</b> library to fine tune the parameters for the SVM model. After running SVM I got gamma = 1, cost = 10 and epsilon = 0.1.

## Predictions

The SVM's tune operation is expensive as it takes long times to run but produces better results.

Predictions done by SVM are as follows:




```r
as.factor(predictions)
```

```
##  [1] B A B A A E C B A A B C B A D D A B B B
## Levels: A B C D E
```
## Conclusion

It is found that Random Forest has done pretty well in predicting the classe in the test data with the accuracy of 95%. Also, SVM has done quite well with MSE 0.20 but as discussed earlier SVM's tune function took a toll on the machine.
