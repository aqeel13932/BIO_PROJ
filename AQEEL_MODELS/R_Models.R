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
x_train<- traindata[ 51:9285, !(names(traindata) %in% drops)]
y_train<- traindata[51:9285,11:18]
x_test<- traindata[ 1:50, !(names(traindata) %in% drops)]
y_test<-traindata[1:50,11:18]

set.seed(825)

#install.packages("drat", repos="https://cran.rstudio.com")
#drat:::addRepo("dmlc")
#install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")
data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
train <- agaricus.train
test <- agaricus.test
dim(x_train)
dim(y_train)
dtrain<-xgb.DMatrix(data= data.matrix(x_train),label = data.matrix(y_train))
require(xgboost)
bstSparse <- xgboost(data =data.matrix(x_train), label = data.matrix(y_train), max_depth = 10, eta = 1, nthread = 2, nrounds = 2, objective = "multi:softprob")
?xgboost
