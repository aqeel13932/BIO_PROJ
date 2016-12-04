# Get the arguments
myArgs <- commandArgs(trailingOnly = TRUE)

# Convert to numerics
nums = as.numeric(myArgs)
dim(nums)<-c(8,100)
nums<- t(nums)
prediction<-nums[1:50,]
original<-nums[51:100,]

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
print(evalGold(prediction,original))