library(laGP)

ndim <- 5

cur_opt <- read.csv("mic_gp.archive.obs_pop.csv")
cur_opt <- cur_opt$func


#get training data points
training_dv <- read.csv("mic_gp.training.dv_pop.csv")
training_obs <- read.csv("mic_gp.training.obs_pop.csv")

training_data <- cbind(training_dv, training_obs)
#training_data <- read.csv("trainingdata.csv")

#set training data for objectives
for (i in 1:ndim) {
  if (i == 1)  X <- training_data[,i+1]
  else X <- cbind(X, training_data[,i+1])
  }

Y <- training_data$func

#get untried location
dv <- read.csv("dv.dat", header=T)
Xref_obj <- matrix(unlist(dv),nrow=1)

f1.alc <- laGP(Xref_obj, round(dim(X)[1]/2), round(2*dim(X)[1]/3), X, Y, d=NULL, method = "alc")

I <- cur_opt-f1.alc$mean
Z <- I/sqrt(f1.alc$s2)

if (f1.alc$s2 > 0){
  EI <- I*pnorm(Z) + dnorm(Z)*sqrt(f1.alc$s2)
} else {
  EI <-0
}

output <- c(f1.alc$mean, sqrt(f1.alc$s2), EI)

write.csv(output, "output.dat", quote = F, row.names = F)
