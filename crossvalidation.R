# fine tuning
test = read.csv("testdataset_v8c.csv",header=TRUE, colClasses = "numeric")
train = read.csv("traindataset_v8c.csv",header=TRUE, colClasses = "numeric")
predictors = names(train)[c(-1,-2)]
train$TripType = as.numeric(as.character(train$TripType))

train = train[1:60000,]

set.seed(123)
x = sample(1:40000, 20000)
set.seed(1234)
y = sample(1:40000, 20000)

err= 100000
eta = 0.3
#  depth =50
#  round = 20
  for (depth in seq(1,20,1)) {
    for (rounds in seq(1,100,1)) {
      
      # train
      bst <- xgboost(data = as.matrix(train[1:40000,predictors]),
                     label = train$TripType[1:40000],
                     max.depth=4, nround=1000, eta=0.15,objective = "multi:softprob", 
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
      err2
      gc()
      
      if (err2 < err) {
        err = err2
        print(paste(depth,rounds,eta,err))
        answer = c(depth,rounds,eta,err)
        write.csv(answer,"result.csv", row.names = FALSE)
      }     
    }
  }  
