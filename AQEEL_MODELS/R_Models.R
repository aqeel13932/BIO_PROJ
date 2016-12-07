rm(list=ls())
calc.cor.test <- function(x=NULL,y=NULL) {
  cor.res <- cor.test(x,y,method = 'spear',alternative = 'great', continuity = T)
  res <- c(cor.res$est,cor.res$p.val)
  names(res) <- c('rho','p.val')
  return(res)
}

evalGold <- function(pred=NULL,goldStandard=NULL) {
  col.eval <- apply(as.array(1:8),1,function(x) calc.cor.test(x=pred[,x],y=goldStandard[,x]))
  colnames(col.eval) <- colnames(pred)
  col.eval['p.val',col.eval['p.val',] < 1e-16] <- 1e-16
  row.eval <- apply(as.array(1:50),1,function(x) calc.cor.test(x=rank(pred[x,],ties.method = 'min'),y=rank(goldStandard[x,],ties.method = 'min')))
  colnames(row.eval) <- rownames(pred)
  row.eval['p.val',row.eval['p.val',] < 1e-16] <- 1e-16
  geom.mean.col <- exp(sum(log(col.eval['p.val',]))/8)
  geom.mean.row <- exp(sum(log(row.eval['p.val',]))/50)
  score <- sum(-log(c(geom.mean.row,geom.mean.col)))/2
  return(score)
}
rankmat <- function(mat=NULL){
mat = apply(mat,2,order,decreasing = TRUE)
return(mat)
}

setwd ('/home/aqeel/Documents/3rdsemester/BIO_PROJ/AQEEL_MODELS/')
traindata<- read.csv('../Data/train.csv')
ftest<- read.csv('../Data/test.csv')
drops<- c("GeneID","wt.t.0","tfA.del.t.0","tfA.del.t.10" ,"tfA.del.t.20" ,"tfA.del.t.30" ,"tfA.del.t.45" ,"tfA.del.t.60" ,"tfA.del.t.90" ,"tfA.del.t.120")
parameters<- c("absolute.expression..parental.strain.t.0..arbitrary.units.","wt.t.10","wt.t.20","wt.t.30","wt.t.45","wt.t.60","wt.t.90","wt.t.120","tfB.del.t.0" ,"tfB.del.t.10","tfB.del.t.20","tfB.del.t.30","tfB.del.t.45","tfB.del.t.60","tfB.del.t.90","tfB.del.t.120","tfC.del.t.0" ,"tfC.del.t.10","tfC.del.t.20","tfC.del.t.30","tfC.del.t.45","tfC.del.t.60","tfC.del.t.90","tfC.del.t.120" )
require(xgboost)
#normalize data
traindata = (traindata- min(traindata))/(max(traindata)-min(traindata))
#traindata = traindata*100
x_train<- traindata[ 51:9285, !(names(traindata) %in% drops)]
dtrain1<-xgb.DMatrix(data= data.matrix(x_train),label = data.matrix(traindata[51:9285,11]))
dtrain2<-xgb.DMatrix(data= data.matrix(x_train),label = data.matrix(traindata[51:9285,12]))
dtrain3<-xgb.DMatrix(data= data.matrix(x_train),label = data.matrix(traindata[51:9285,13]))
dtrain4<-xgb.DMatrix(data= data.matrix(x_train),label = data.matrix(traindata[51:9285,14]))
dtrain5<-xgb.DMatrix(data= data.matrix(x_train),label = data.matrix(traindata[51:9285,15]))
dtrain6<-xgb.DMatrix(data= data.matrix(x_train),label = data.matrix(traindata[51:9285,16]))
dtrain7<-xgb.DMatrix(data= data.matrix(x_train),label = data.matrix(traindata[51:9285,17]))
dtrain8<-xgb.DMatrix(data= data.matrix(x_train),label = data.matrix(traindata[51:9285,18]))

x_test<- traindata[ 1:50, !(names(traindata) %in% drops)]
dtest1<-xgb.DMatrix(data= data.matrix(x_test),label = data.matrix(traindata[1:50,11]))
dtest2<-xgb.DMatrix(data= data.matrix(x_test),label = data.matrix(traindata[1:50,12]))
dtest3<-xgb.DMatrix(data= data.matrix(x_test),label = data.matrix(traindata[1:50,13]))
dtest4<-xgb.DMatrix(data= data.matrix(x_test),label = data.matrix(traindata[1:50,14]))
dtest5<-xgb.DMatrix(data= data.matrix(x_test),label = data.matrix(traindata[1:50,15]))
dtest6<-xgb.DMatrix(data= data.matrix(x_test),label = data.matrix(traindata[1:50,16]))
dtest7<-xgb.DMatrix(data= data.matrix(x_test),label = data.matrix(traindata[1:50,17]))
dtest8<-xgb.DMatrix(data= data.matrix(x_test),label = data.matrix(traindata[1:50,18]))


#install.packages("drat", repos="https://cran.rstudio.com")
#drat:::addRepo("dmlc")
#install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")
md=6
eta=1
nthrd=2
nronds=10

set.seed(825)
mod1 <- xgboost(booster = "gblinear",data =dtrain1, max_depth = md, eta = eta, nthread =nthrd, nrounds = nronds, objective = "reg:linear")
mod2 <- xgboost(booster = "gblinear",data =dtrain2, max_depth = md, eta = eta, nthread =nthrd, nrounds = nronds, objective = "reg:linear")
mod3 <- xgboost(booster = "gblinear",data =dtrain3, max_depth = md, eta = eta, nthread =nthrd, nrounds = nronds, objective = "reg:linear")
mod4 <- xgboost(booster = "gblinear",data =dtrain4, max_depth = md, eta = eta, nthread =nthrd, nrounds = nronds, objective = "reg:linear")
mod5 <- xgboost(booster = "gblinear",data =dtrain5, max_depth = md, eta = eta, nthread =nthrd, nrounds = nronds, objective = "reg:linear")
mod6 <- xgboost(booster = "gblinear",data =dtrain6, max_depth = md, eta = eta, nthread =nthrd, nrounds = nronds, objective = "reg:linear")
mod7 <- xgboost(booster = "gblinear",data =dtrain7, max_depth = md, eta = eta, nthread =nthrd, nrounds = nronds, objective = "reg:linear")
mod8 <- xgboost(booster = "gblinear",data =dtrain8, max_depth = md, eta = eta, nthread =nthrd, nrounds = nronds, objective = "reg:linear")

pred1<-predict(mod1,data.matrix(x_test))
pred2<-predict(mod2,data.matrix(x_test))
pred3<-predict(mod3,data.matrix(x_test))
pred4<-predict(mod4,data.matrix(x_test))
pred5<-predict(mod5,data.matrix(x_test))
pred6<-predict(mod6,data.matrix(x_test))
pred7<-predict(mod7,data.matrix(x_test))
pred8<-predict(mod8,data.matrix(x_test))
pred1

pred<- cbind(pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8)
evalGold(rankmat(traindata[1:50,11:18]),rankmat(traindata[1:50,11:18]))
evalGold(rankmat(pred),rankmat(traindata[1:50,11:18]))
traindata$tfA.del.t.0[1:50]

sqrt(sum( (pred1 -traindata[1:50,11] )^2 , na.rm = TRUE ) / nrow(x_test))
