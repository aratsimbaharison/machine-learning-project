Executive summary
-----------------

The goal of your project is to predict the manner in which the
participants did their exercise (correctly or incorrectly). This is the
"classe" variable in the training set.

Two prediction models, classification trees and random forest, are
compared in order to find out which model performs best in predicting
the manner in which the participants did their exercise.

This report describe how the prediction models were built, and how cross
validation was conducted.

As a result, this project shows that the random forest model performs
better that the classification trees in predicting the manner in which
the participants did their exercise.

Background
----------

Using devices such as *Jawbone Up, Nike FuelBand,* and *Fitbit* it is
now possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement - a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways. More information is
available from the website here:
<http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight
Lifting Exercise Dataset).

Loading and cleaning the data
-----------------------------

    # import the data from the URLs

    # trainurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    # testurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    # training <- download.file(trainurl, "training.csv")
    # testing <- download.file(testurl, "testing.csv")

    # load the data locally
    training <- read.csv("training.csv", na.strings = c("NA", ""))
    testing <- read.csv("testing.csv", na.strings = c("NA", ""))

    # deleting columns (predictors) that contain any missing values
    training <- training[, colSums(is.na(training)) == 0]
    testing <- testing[, colSums(is.na(testing)) == 0]

Splitting the training data set for out-of-sample errors
--------------------------------------------------------

In order to get out-of-sample errors, we need to split the training set
into train data set (train, 70%) for prediction and a validation data
set (valid 30%)

    # splitting the training data set:

    set.seed(123) 
    inTrain <- createDataPartition(training$classe, p = 0.7, list = FALSE)
    train <- training[inTrain, ]
    valid <- training[-inTrain, ]

Using two prediction models: classification trees and random forest
-------------------------------------------------------------------

### Classification trees model

We use here a 5-fold cross validation (default setting in trainControl
function is 10) to save a computing time.

    control <- trainControl(method = "cv", number = 5)
    fit_rpart <- train(classe ~ ., data = train, method = "rpart", 
                       trControl = control)
    print(fit_rpart, digits = 4)

    ## CART 
    ## 
    ## 13737 samples
    ##    59 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 10989, 10991, 10989, 10990, 10989 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp      Accuracy  Kappa 
    ##   0.2437  0.7662    0.7029
    ##   0.2568  0.5513    0.4263
    ##   0.2704  0.3617    0.1325
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was cp = 0.2437.

    # plotting the classification trees
    fancyRpartPlot(fit_rpart$finalModel)

![](machine-learning-project_files/figure-markdown_strict/unnamed-chunk-4-1.png)

    # predicting the outcomes with the classification trees model using the validation data set
    predict_rpart <- predict(fit_rpart, valid)
    # Showing the confusion matrix
    confusionMatrix(predict_rpart, valid$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1674    0    0    0    0
    ##          B    0 1139    0    0    0
    ##          C    0    0    0    0    0
    ##          D    0    0    0    0    0
    ##          E    0    0 1026  964 1082
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.6619          
    ##                  95% CI : (0.6496, 0.6739)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.5696          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   1.0000   0.0000   0.0000   1.0000
    ## Specificity            1.0000   1.0000   1.0000   1.0000   0.5857
    ## Pos Pred Value         1.0000   1.0000      NaN      NaN   0.3522
    ## Neg Pred Value         1.0000   1.0000   0.8257   0.8362   1.0000
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2845   0.1935   0.0000   0.0000   0.1839
    ## Detection Prevalence   0.2845   0.1935   0.0000   0.0000   0.5220
    ## Balanced Accuracy      1.0000   1.0000   0.5000   0.5000   0.7928

As we can see under the overall statistics, the accuracy of the
classification trees model is: 0.6619

### Random forest Model

Now we try the random forest model:

    fit_rf <- train(classe ~ ., data = train, method = "rf", 
                       trControl = control)
    print(fit_rf, digits = 4)

    ## Random Forest 
    ## 
    ## 13737 samples
    ##    59 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 10990, 10989, 10991, 10989, 10989 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy  Kappa 
    ##    2    0.9939    0.9923
    ##   41    0.9999    0.9999
    ##   81    0.9997    0.9996
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was mtry = 41.

    # predicting the outcomes with the random forest model using the validation data set
    predict_rf <- predict(fit_rf, valid)

    # Showing the confusion matrix of the random forest model
    confusionMatrix(predict_rf, valid$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1674    0    0    0    0
    ##          B    0 1139    1    0    0
    ##          C    0    0 1025    0    0
    ##          D    0    0    0  964    0
    ##          E    0    0    0    0 1082
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 0.9998     
    ##                  95% CI : (0.9991, 1)
    ##     No Information Rate : 0.2845     
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 0.9998     
    ##  Mcnemar's Test P-Value : NA         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   1.0000   0.9990   1.0000   1.0000
    ## Specificity            1.0000   0.9998   1.0000   1.0000   1.0000
    ## Pos Pred Value         1.0000   0.9991   1.0000   1.0000   1.0000
    ## Neg Pred Value         1.0000   1.0000   0.9998   1.0000   1.0000
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2845   0.1935   0.1742   0.1638   0.1839
    ## Detection Prevalence   0.2845   0.1937   0.1742   0.1638   0.1839
    ## Balanced Accuracy      1.0000   0.9999   0.9995   1.0000   1.0000

As we can see under the overall statistics, the accuracy of the random
forest model is: 0.9998.

Prediction with the random forest model on the testing data set
---------------------------------------------------------------

Now we use the random forest model on the testing data set to see if it
still performs better than the classification trees model.

    predict_rfTest <- predict(fit_rf, testing)
    predict_rfTest

    ##  [1] A A A A A A A A A A A A A A A A A A A A
    ## Levels: A B C D E

Conclusion
----------

This project shows that the random forest model performs better that the
classification trees in predicting the manner in which the participants
did their exercise.
