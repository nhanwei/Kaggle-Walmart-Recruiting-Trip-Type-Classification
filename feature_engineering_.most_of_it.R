# segmenting their store visits into different trip types
# 38 distinct types

# The training set (train.csv) contains a large number of customer visits with the TripType included. 
# You must predict the TripType for each customer visit in the test set (test.csv).

# TripType - a categorical id representing the type of shopping trip the customer made. 
# This is the ground truth that you are predicting. TripType_999 is an "other" category.
# VisitNumber - an id corresponding to a single trip by a single customer
# Weekday - the weekday of the trip
# Upc - the UPC number of the product purchased
# ScanCount - the number of the given item that was purchased. A negative value indicates a product return.
# DepartmentDescription - a high-level description of the item's department
# FinelineNumber - a more refined category for each of the products, created by Walmart

# > str(train)
# 'data.frame':	647054 obs. of  7 variables:
# $ TripType             : int  999 30 30 26 26 26 26 26 26 26 ...
# $ VisitNumber          : int  5 7 7 8 8 8 8 8 8 8 ...
# $ Weekday              : Factor w/ 7 levels "Friday","Monday",..: 1 1 1 1 1 1 1 1 1 1 ...
# $ Upc                  : num  6.81e+10 6.05e+10 7.41e+09 2.24e+09 2.01e+09 ...
# $ ScanCount            : int  -1 1 1 2 2 2 1 1 1 -1 ...
# $ DepartmentDescription: Factor w/ 69 levels "1-HR PHOTO","ACCESSORIES",..: 21 64 52 51 51 51 51 51 51 51 ...
# $ FinelineNumber       : int  1000 8931 4504 3565 1017 1017 1017 2802 4501 3565 ...

### loading libraries
library(xgboost)
library(reshape2)
library(caret)
library(Metrics)
library(RCurl) # download https data

### reading in data
sample_sub = read.csv("sample_submission.csv", header=TRUE)
testbase = read.csv("test.csv", header=TRUE)
trainbase = read.csv("train.csv", header=TRUE)

dataset = data.frame(unique(train[,c(1,2)]))    # origin dataset with unique visitnumber and triptype pair
traindmy = dummyVars(" ~ .", data=train, fullRank = FALSE) # creates dummy variable
traindmy2 = data.frame(predict(traindmy, newdata=train))# "flatten" the factor variables

############### feature creation for department description and day of the week. changing it to dummy
DD_names = names(traindmy2)[c(3:9,12:80)] # creating new features for DepartmentDescription (DD)
DD_names2 = matrix(rep(0, 76*95674),95674,76) # all zeros for DD first
dataset = data.frame(dataset, DD_names2) # combine with origin empty datasets
names(dataset)[3:78] =  DD_names    # giving correct names for the DD
for (i in 1:dim(dataset)[1]){
  hitlist = traindmy2[traindmy2$VisitNumber==dataset[i,2], ]
  dayofweek = names(hitlist)[3:9][hitlist[1,3:9]>0]       # taking the day of week out
  dataset[i,dayofweek] = 1      # populating the day of week columns
  
  hitlist2 = train[train$VisitNumber == dataset[i,2],]
  agg = aggregate(ScanCount~DepartmentDescription, data=hitlist2,sum)
  sub1 = gsub(" ", ".",paste0(names(agg)[1], ".",agg[,1]))
  sub2 = gsub("-", ".", sub1)
  sub3 = gsub("&", ".",sub2)
  sub4 = gsub("/", ".",sub3)
  sub5 = gsub(",", ".",sub4)
  dataset[i,sub5] = agg[,2]
  print(i)
}
write.csv(dataset, "traindataset_v1.csv", row.names = FALSE)

############### feature creation for department description and day of the week. changing it to dummy
testdataset = data.frame(unique(test[,c(1)]))    # origin dataset with unique visitnumber and triptype pair
testdmy = matrix(rep(0, 76*95674),95674,76)
testdmy = data.frame(testdmy)
names(testdmy) = names(dataset)[3:78]
testdataset = data.frame(testdataset,testdmy)
names(testdataset)[1]= "VisitNumber"
testdmy2 = dummyVars(" ~ .", data=test, fullRank = FALSE) # creates dummy variable
testdmy3 = data.frame(predict(testdmy2, newdata=test))# "flatten" the factor variables

for (i in 1:dim(testdataset)[1]){
  hitlist = testdmy3[testdmy3$VisitNumber==testdataset[i,1], ]
  dayofweek = names(hitlist)[2:8][hitlist[1,2:8]>0]       # taking the day of week out
  testdataset[i,dayofweek] = 1      # populating the day of week columns
  
  hitlist2 = test[test$VisitNumber == testdataset[i,1],]
  agg = aggregate(ScanCount~DepartmentDescription, data=hitlist2,sum)
  sub1 = gsub(" ", ".",paste0(names(agg)[1], ".",agg[,1]))
  sub2 = gsub("-", ".", sub1)
  sub3 = gsub("&", ".",sub2)
  sub4 = gsub("/", ".",sub3)
  sub5 = gsub(",", ".",sub4)
  testdataset[i,sub5] = agg[,2]
  print(i)
}
write.csv(testdataset, "testdataset_v1.csv", row.names = FALSE)

######################################################################
testdata = read.csv("testdataset_v1.csv", header=TRUE)
traindata = read.csv("traindataset_v1.csv", header=TRUE)
sample_sub = read.csv("sample_submission.csv", header=TRUE)
test = read.csv("test.csv", header=TRUE)
train = read.csv("train.csv", header=TRUE)

#### creating "gotReturn" feature
trainReturn = train[train$ScanCount<0,]
testReturn = test[test$ScanCount<0,]
trainReturn = unique(trainReturn$VisitNumber)
testReturn = unique(testReturn$VisitNumber)
trainReturn = data.frame("gotReturn" = 1, "VisitNumber"= trainReturn )
testReturn = data.frame("gotReturn" = 1, "VisitNumber"= testReturn)

traindata2 = merge(traindata, trainReturn, by="VisitNumber", all.x=TRUE)
testdata2 = merge(testdata, testReturn, by="VisitNumber", all.x=TRUE)
traindata2[is.na(traindata2$gotReturn),79] = 0
testdata2[is.na(testdata2$gotReturn),78] = 0


####### mapping function

mapping = data.frame("TripType" = names(table(trainSet$TripType)), "mapped" = 0:37)
for (i in 1:dim(trainSet)[1]){
  trainSet$TripType[i] = mapping[mapping$TripType==trainSet$TripType[i], 2]
}
dataset$TripType = as.numeric(as.character(dataset$TripType))
for (i in 3:dim(dataset)[2]){
  dataset[,i] = as.numeric(dataset[,i])
}


########### creating finelinenumber feature
train2 = train[,c(2,5,7)]
train2$FinelineNumber[is.na(train2$FinelineNumber)] =10000
unique_vis = unique(train2$VisitNumber)
fln = unique(train2$FinelineNumber)
fln2 = sort(fln)
fln3 = paste0("fln", fln2)

traindataset = data.frame("VisitNumber"= unique_vis)
newcol = rep(0,95674)
for(i in 1:5196){
  traindataset = cbind(traindataset,newcol)
}
hold = traindataset[,1]
traindataset = traindataset[,-1]
names(traindataset)= fln3

for(i in 1:647054){
  traindataset[unique_vis==train2[i,1],fln2==train2[i,3]]= traindataset[unique_vis==train2[i,1],fln2==train2[i,3]]+ train2[i,2]
}

traindataset = data.frame("VisitNumber"= unique_vis, traindataset)
traindata3 = merge(traindata2, flnmat, by="VisitNumber", all.x=TRUE)
sum(is.na(traindata3[,1224]))
write.csv(traindataset, "traindataset_v2.csv", row.names = FALSE)




