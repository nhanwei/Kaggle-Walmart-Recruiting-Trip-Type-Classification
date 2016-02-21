############ ensemble script two
prednames = names(sample_sub)[2:39]
rm(sample_sub)
#names(pred)[2:39] =names(sample_sub)[2:39]
hold = read.csv("submission.csv", header=TRUE)

count = 1
x = rep(0,360)
y = rep(0,360)
#x= read.csv("x.csv")
#y = read.csv("y.csv")
errkeep =rep(0,360)

count2 = 1

## stopped at 800
#for (depth in 3:20){
depth = 9
#round= 500
for (depth in 7:20){
    for (round in seq(400,800, 100)){
        #train
        bst <- xgboost(data = as.matrix(train[1:40000,predictors]),
                       label = train$TripType[1:40000],
                       max.depth=depth, nround=round, eta=0.15,objective = "multi:softprob", 
                       num_class = 38, eval_metric = "mlogloss", subsample=1, 
                       colsample_bytree = 0.1, min_child_weight = 1, gamma = 0.6, max_delta_step =10)
        gc()
        
        # predict
        predictions <- predict(bst, as.matrix(train[40001:95674,predictors]))
        predictions =  matrix(predictions, 38, 55674, byrow=FALSE)
        predictions = t(predictions)
        predictions = data.frame(cbind(train[40001:95674,"TripType"], predictions))
        err2=-sum(log(apply(predictions,1, function(x){
            y=x[1] +2
            return(x[y])
        })))
        print(err2)
        errkeep[count2] = err2
        gc()
        ### if err2 < 45000, we start doing the predictions on test dataset
        if (err2 < 45000){
            rm(predictions)
            rm(bst)
            gc()
            bst <- xgboost(data = as.matrix(train[,predictors]),
                           label = train[,2],
                           max.depth=depth, nround=round, eta=0.15,objective = "multi:softprob", 
                           num_class = 38, eval_metric = "mlogloss", subsample=1, 
                           colsample_bytree = 0.1, min_child_weight= 1, gamma= 0.6, max_delta_step=10)
            gc()
            pred <- predict(bst, as.matrix(test[,predictors]))
            pred = matrix(pred, 38, 95674, byrow=FALSE)
            pred = t(pred)
            pred = data.frame("VisitNumber"= test$VisitNumber, pred)
            names(pred)[2:39] =prednames
            pred[,1] = as.integer(pred[,1])
            if (count == 1){
                hold = pred
                count = count+1
            } else {
                hold = (hold*(count-1)+pred)/count
                count = count + 1
            }
            rm(pred)
            rm(bst)
            gc()
            break
        }
        x[count2] = depth
        y[count2] = round
        count2 = count2 + 1
        write.csv(hold, "submissionensemble2.csv", row.names = FALSE)
        write.csv(x, "xensemble2.csv", row.names = FALSE)
        write.csv(y, "yensemble2.csv", row.names = FALSE)
        write.csv(count, "countensmble2.csv", row.names = FALSE)
        write.csv(count2, "count2ensemble2.csv", row.names = FALSE)
    }
}



