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

#x = 1:50
#xs = cbind(c(1,(sample(x[2:50],49,replace = FALSE))),c(1,(sample(x[2:50],49,replace = FALSE))),c(1,(sample(x[2:50],49,replace = FALSE))),c(1,(sample(x[2:50],49,replace = FALSE))),c(1,(sample(x[2:50],49,replace = FALSE))),c(1,(sample(x[2:50],49,replace = FALSE))),c(1,(sample(x[2:50],49,replace = FALSE))),c(1,(sample(x[2:50],49,replace = FALSE))))
#dim(xs)
#xs
#evalGold(xs,xs)
#warnings()
setwd ('/home/aqeel/Documents/3rdsemester/BIO_PROJ/AQEEL_MODELS/')
traindata<- read.csv('../Data/train.csv')
ftest<- read.csv('../Data/test.csv')
drops<- c("GeneID","wt.t.0","tfA.del.t.0","tfA.del.t.10" ,"tfA.del.t.20" ,"tfA.del.t.30" ,"tfA.del.t.45" ,"tfA.del.t.60" ,"tfA.del.t.90" ,"tfA.del.t.120")
parameters<- c("absolute.expression..parental.strain.t.0..arbitrary.units.","wt.t.10","wt.t.20","wt.t.30","wt.t.45","wt.t.60","wt.t.90","wt.t.120","tfB.del.t.0" ,"tfB.del.t.10","tfB.del.t.20","tfB.del.t.30","tfB.del.t.45","tfB.del.t.60","tfB.del.t.90","tfB.del.t.120","tfC.del.t.0" ,"tfC.del.t.10","tfC.del.t.20","tfC.del.t.30","tfC.del.t.45","tfC.del.t.60","tfC.del.t.90","tfC.del.t.120" )
require(xgboost)
#normalize data
#traindata = (traindata- min(traindata))/(max(traindata)-min(traindata))
#traindata = traindata*100
x_train<- traindata[ 51:9285, !(names(traindata) %in% drops)]
data.matrix(traindata[51:9285,11])
dim(x_train)
dim(data.matrix(traindata[51:9285,11:18]))
traindata[51:9285,11:18]
dim(x_train)
dim(traindata[51:9285,11:18])
xgb.DMatrix(data=data.matrix(x_train),label = traindata[51:9285,11:18])
dtrain1<-xgb.DMatrix(data= data.matrix(x_train),label = traindata[51:9285,11:18])
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


min(traindata)
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


x<- rbind(c( 32,39,39,38,37,30,31,27),
      c( 35,29,34,32,33,35,34,32),
      c( 41,20,16,19,19,19,29,30),
      c( 43,50,50,50,50,50,50,50),
      c( 50,42,45,43,44,41,40,35),
      c(2,33,24, 8, 5, 4, 8, 7),
      c( 34,36,35,34,32,29,32,31),
      c( 25,35, 9, 3, 3, 2, 3, 4),
      c( 38,40,38,35,35,39,45,45),
      c( 44,48,48,49,45,38,42,40),
      c( 45,41,41,45,43,44,44,44),
      c(9, 4, 2, 2, 2, 3,10,17),
      c( 17,19,14,12, 9,11,16,18),
      c( 28, 7, 4, 6, 6,15,25,33),
      c( 20,24,19,20,20,24,24,28),
      c( 46,34,37,39,42,47,47,46),
      c( 30,26,25,23,24,18,22,21),
      c( 27,27,31,30,28,27,26,25),
      c(8, 2, 3, 5, 7,10, 9, 9),
      c( 33,45,40,40,41,37,38,36),
      c( 11, 3,11,27,27,28,18,24),
      c( 13,18,22,21,18,13,14,11),
      c( 24, 9,10,16,12, 6, 5, 6),
      c( 19,10, 8,15,13, 5, 4, 5),
      c( 37, 8,15,28,29,33,35,39),
      c( 15,23,27,26,25,25,20,20),
      c(3,13,17,17,17, 9, 6, 2),
      c( 47,30,33,37,39,46,46,47),
      c( 10,28,28,24,26,26,17,22),
      c( 12,21,18,18,22,21,15,19),
      c( 48,47,43,44,40,48,48,48),
      c( 49,44,42,47,47,49,49,49),
      c( 21,16,21,14,14,14,19,12),
      c( 23, 5, 5, 9, 8,12,11,10),
      c( 29,46,46,46,48,36,28,16),
      c( 26,49,49,42,34,34,33,34),
      c( 39,38,36,36,38,40,41,43),
      c( 40, 1, 1, 1, 1, 1, 1, 1),
      c(1,32,32,31,30,32,37,37),
      c( 16,15,20,13,15,16,23,14),
      c( 31,11,26,29,31,31,30,29),
      c( 18,31,29,25,23,23,27,26),
      c(5,17,12, 7,11,17,12,13),
      c( 22,37,44,41,46,43,39,41),
      c(4, 6, 6,10,10, 8, 2, 3),
      c( 14,25,23,22,21,22,21,23),
      c( 36,43,47,48,49,45,43,42),
      c(7,12, 7, 4, 4, 7, 7, 8),
      c(6,14,13,11,16,20,13,15),
      c( 42,22,30,33,36,42,36,38))
y<-rbind(c( 14,40,43,40,43,21,44,12),
     c( 40,42,36,34,42,34,32,36),
     c( 22,22,14,21,32,24,34,34),
     c( 50,50,50,50,50,50,50,50),
     c( 15,12,42,47,48,43,48,39),
     c(  1,48, 7, 3,10, 7,22,13),
     c( 36,38,39,35,37,31,29,33),
     c( 26,49, 3, 2, 4, 5,12, 9),
     c( 41,36,33,30,26,39,49,46),
     c( 49,47,44,48,28,23,45,37),
     c( 47,33,37,38,38,36,47,42),
     c(  7, 6, 2, 4, 2, 3,13,23),
     c( 25,26,10,13,12,13,30,35),
     c(  2,11, 6,15, 5, 8,38,30),
     c( 17,17,21, 6, 8, 6,19,25),
     c( 32, 5, 5,17,19,28,26,44),
     c( 38,10,23,22,18,11,16,19),
     c( 35,27,28,28,31,17,15,29),
     c( 16, 1,12, 8,22,20,21, 8),
     c( 37,39,40,44,45,47,36,41),
     c( 11, 7,38,36,41,42,35,27),
     c( 23,28,25,26,17,10,14, 5),
     c( 21,18,19,18,15, 2, 6, 4),
     c( 27,14,15,14, 9, 4, 5, 2),
     c( 30,13, 9,10,14,30,31,10),
     c( 13,25,26,31,13,29,24,14),
     c(  4, 9,11,32,40,41, 7, 6),
     c( 39, 4, 4,37,33,32,43,45),
     c( 18,32,31,29,34,38,17,26),
     c( 33,21,35,27,29,26,23,24),
     c( 48,37,46,41,24,33,37,40),
     c( 44,45,48,39,35,44,41,38),
     c( 42,15,34, 5, 3,15,10,11),
     c( 31,34,22,19,11,14, 4,15),
     c( 20,41,41,46,46,40,20,22),
     c(  3,46,49,45,30,35,42,43),
     c( 43,44,16,33,39,46,33,48),
     c(  5, 2, 1, 1, 1, 1, 1, 1),
     c( 24,20,20,12, 6,22,28,31),
     c( 34,29,32,23,21,16,25,16),
     c( 29, 8,18,25,16,27,18,28),
     c(  8,24,29,24,36,37,27,21),
     c( 28,30,30, 9,25, 9, 8,17),
     c(  9,35,45,42,47,45,39,32),
     c( 10,23,17,16,27,18, 3, 3),
     c( 19,16, 8,11,20,19,11,20),
     c( 46,43,47,49,49,49,46,49),
     c(  6,19,13, 7, 7,12, 2, 7),
     c( 12,31,24,20,23,25, 9,18),
     c( 45, 3,27,43,44,48,40,47))

evalGold(x,y)
