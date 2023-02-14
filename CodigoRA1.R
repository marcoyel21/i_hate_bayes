### ----- REGRESION AVANZADA ----- ###
# --- Prof. Luis E. Nieto Barajas --- #


#--- Usar espejo CRAN del ITAM ---
options(repos="http://cran.itam.mx/")

#--- Funciones utiles ---
prob<-function(x){
  out<-min(length(x[x>0])/length(x),length(x[x<0])/length(x))
  out
}

#--- Ilustracion del proceso de inferencia ---

#-Proceso de aprendizaje normal-normal-
xbar<-40.9533
sig2<-4
n<-3

th0<-39
sig20<-219.47

y<-seq(35,45,length.out=200)
f0y<-dnorm(y,th0,sqrt(sig20))
liky<-dnorm(y,xbar,sqrt(sig2/n))
sig21<-1/(n/sig2+1/sig20)
th1<-sig21*(n/sig2*xbar+th0/sig20)
f1y<-dnorm(y,th1,sqrt(sig21))
ymax<-max(f0y,liky,f1y)
plot(y,f0y,ylim=c(0,ymax),type="l")
lines(y,liky,lty=2,col=2)
lines(y,f1y,lty=3,col=3)

#-Proceso de aprendizaje bernoulli-beta-

#Simulacion de datos Bernoulli
theta0 <- 0.6
n <- 100
x<-rbinom(n,1,theta0)
hist(x,freq=FALSE)

#Distribucion inicial para theta
a <- 1
b <- 1
theta<-seq(0,1,,100)
plot(theta,dbeta(theta,a,b),type="l")

#Distribucion final
a1 <- a + sum(x)
b1 <- b + n - sum(x)
plot(theta,dbeta(theta,a1,b1),type="l")

#Ambas
theta<-seq(0,1,,100)
ymax <- max(dbeta(theta,a,b),dbeta(theta,a1,b1))
plot(theta,dbeta(theta,a,b),type="l",ylim=c(0,ymax))
lines(theta,dbeta(theta,a1,b1),col=2)
abline(v=theta0,col=4)

#Aproximacion normal asintotica
mu <- (a1-1)/(a1+b1-2)
sig2 <- (a1-1)*(b1-1)/(a1+b1-2)^3
lines(theta,dnorm(theta,mu,sqrt(sig2)),col=3)


# --- Aproximación Monte Carlo --- 
#-Ejemplo 1-
x<-seq(-2,4,,1000)
f<-function(x){
  out <- 5-(x-1)^2
  out <- ifelse (x < -1 | x>3,0,out)
  out
}
plot(x,f(x)*3/44,type="l",ylim=c(0,0.5))
lines(x,dnorm(x,0,1),lty=2,col=2)
lines(x,dnorm(x,1,2/3),lty=3,col=3)
lines(x,dnorm(x,1,1),lty=4,col=4)
lines(x,dnorm(x,1,2),lty=5,col=5)

N<-100

#Caso 1: S=Normal estandar
mu<-0
sig<-1
y<-rnorm(N,mu,sig)
I1<-mean(f(y)/dnorm(y,mu,sig))
eI1<-sd(f(y)/dnorm(y,0,1))/sqrt(N)
print(c(I1,eI1))

#Caso 2: S=Normal no estandar
mu<-1
sig<-2/3
y<-rnorm(N,mu,sig)
I2<-mean(f(y)/dnorm(y,mu,sig))
eI2<-sd(f(y)/dnorm(y,mu,sig))/sqrt(N)
print(c(I2,eI2))

#Caso 3: S=Normal no estandar
mu<-1
sig<-1
y<-rnorm(N,mu,sig)
I3<-mean(f(y)/dnorm(y,mu,sig))
eI3<-sd(f(y)/dnorm(y,mu,sig))/sqrt(N)
print(c(I3,eI3))

#Caso 4: S=Normal no estandar
mu<-1
sig<-2
y<-rnorm(N,mu,sig)
I4<-mean(f(y)/dnorm(y,mu,sig))
eI4<-sd(f(y)/dnorm(y,mu,sig))/sqrt(N)
print(c(I4,eI4))


#-Ejemplo 2-
f<-function(x){
  out<-ifelse(x<0,0,x)
  out<-ifelse(x>1,0,out)
  out
}
x<-seq(-1,2,,100)
plot(x,f(x),type="l",ylim=c(0,1.2))
lines(x,dunif(x,0,1),col=2,lty=2)
lines(x,dexp(x,1),col=3,lty=3)
lines(x,dnorm(x,0.5,1/3),col=4,lty=4)

N<-100

#Caso 1: S=Uniforme
y<-runif(N,0,1)
I1<-mean(f(y)/dunif(y,0,1))
eI1<-sd(f(y)/dunif(y,0,1))/sqrt(N)
print(c(I1,eI1))

#Caso 2: S=Exponencial
y<-rexp(N,1)
I2<-mean(f(y)/dexp(y,1))
eI2<-sd(f(y)/dexp(y,1))/sqrt(N)
print(c(I2,eI2))

#Caso 3: S=Normal
y<-rnorm(N,0.5,1/3)
I3<-mean(f(y)/dnorm(y,0.5,1/3))
eI3<-sd(f(y)/dnorm(y,0.5,1/3))/sqrt(N)
print(c(I3,eI3))


#-Muestreador de Gibbs-
install.packages("bayesm")
library(bayesm)
out<-rbiNormGibbs(rho=0.95)
out<-rbiNormGibbs(rho=-0.5)


###########################################

install.packages("R2OpenBUGS")
install.packages("R2jags")
library(R2OpenBUGS)
library(R2jags)

#-Working directory-
wdir<-"c:/temp/RegAva/"
setwd(wdir)


#--- Ejemplo 1---
#-Reading data-
n<-10
credito<-c(rep(1,n/2),rep(0,n/2))
credito<-c(rep(1,n*0.9),rep(0,n*0.1))
credito<-c(rep(0,n*0.9),rep(1,n*0.1))

#-Defining data-
data<-list("n"=n,"x"=credito)

#-Defining inits-
inits<-function(){list(theta=0.5,x1=rep(1,2))}
inits<-function(){list(lambda=0)}
inits<-function(){list(theta=0.5,eta=1)}

#-Selecting parameters to monitor-
parameters<-c("theta","x1")
parameters<-c("theta","eta")

#-Running code-
#OpenBUGS
ej1.sim<-bugs(data,inits,parameters,model.file="Ej1.txt",
              n.iter=5000,n.chains=2,n.burnin=500)
#JAGS
ej1.sim<-jags(data,inits,parameters,model.file="Ej1.txt",
              n.iter=5000,n.chains=2,n.burnin=500,n.thin=1)

#-Monitoring chain-

#Traza de la cadena
traceplot(ej1.sim)

#Cadena

#OpenBUGS
out<-ej1.sim$sims.list

#JAGS
out<-ej1.sim$BUGSoutput$sims.list

z<-out$theta
par(mfrow=c(2,2))
plot(z,type="l")
plot(cumsum(z)/(1:length(z)),type="l")
hist(z,freq=FALSE)
acf(z)

#Resumen (estimadores)
#OpenBUGS
out.sum<-ej1.sim$summary

#JAGS
out.sum<-ej1.sim$BUGSoutput$summary

print(out.sum)

#DIC
#OpenBUGS
out.dic<-ej1.sim$DIC

#JAGS
out.dic<-ej1.sim$BUGSoutput$DIC

print(out.dic)

#---------------------#
#Mezcla de betas
w<-seq(0.01,0.99,,100)
pp<-0.3
fw<-pp*dbeta(w,10,10)+(1-pp)*dbeta(w,5,0.05)
par(mfrow=c(1,1))
plot(w,fw,type="l")


#--- Ejemplo 2---
#-Reading data-
utilidad<-c(212, 207, 210,
196, 223, 193,
196, 210, 202, 221)
n<-length(utilidad)

#-Defining data-
data<-list("n"=n,"x"=utilidad)

#-Defining inits-
inits<-function(){list(mu=0,sig=1,x1=0)}

#-Selecting parameters to monitor-
parameters<-c("mu","sig","x1")

#-Running code-
#OpenBUGS
ej2.sim<-bugs(data,inits,parameters,model.file="Ej2.txt",
              n.iter=5000,n.chains=2,n.burnin=500)
#JAGS
ej2.sim<-jags(data,inits,parameters,model.file="Ej2.txt",
              n.iter=5000,n.chains=2,n.burnin=500,n.thin=1)

#-Monitoring chain-

#Traza de la cadena
traceplot(ej2.sim)

#Cadena

#OpenBUGS
out<-ej2.sim$sims.list

#JAGS
out<-ej2.sim$BUGSoutput$sims.list

z<-out$x1
par(mfrow=c(2,2))
plot(z,type="l")
plot(cumsum(z)/(1:length(z)),type="l")
hist(z,freq=FALSE)
acf(z)

#Resumen (estimadores)
#OpenBUGS
out.sum<-ej2.sim$summary

#JAGS
out.sum<-ej2.sim$BUGSoutput$summary

print(out.sum)

#DIC
#OpenBUGS
out.dic<-ej2.sim$DIC

#JAGS
out.dic<-ej2.sim$BUGSoutput$DIC

print(out.dic)


#--- Ejemplo 3 ---
#-Reading data-
calif<-read.table("http://allman.rhon.itam.mx/~lnieto/index_archivos/calificaciones.txt",header=TRUE)
n<-nrow(calif)
plot(calif$MO,calif$SP)

#-Defining data-
data<-list("n"=n,"y"=calif$SP,"x"=calif$MO)

#-Defining inits-
inits<-function(){list(beta=rep(0,2),tau=1,yf=rep(0,n))}
inits<-function(){list(beta=rep(0,6),tau=1,yf=rep(0,n))}

#-Selecting parameters to monitor-
parameters<-c("beta","tau","yf")

#-Running code-
#OpenBUGS
ej3.sim<-bugs(data,inits,parameters,model.file="Ej3.txt",
              n.iter=10000,n.chains=2,n.burnin=1000)
ej3a.sim<-bugs(data,inits,parameters,model.file="Ej3a.txt",
              n.iter=100000,n.chains=2,n.burnin=10000,n.thin=5)
ej3b.sim<-bugs(data,inits,parameters,model.file="Ej3b.txt",
              n.iter=10000,n.chains=2,n.burnin=1000)
#JAGS
ej3.sim<-jags(data,inits,parameters,model.file="Ej3.txt",
              n.iter=10000,n.chains=2,n.burnin=1000,n.thin=1)
ej3a.sim<-jags(data,inits,parameters,model.file="Ej3a.txt",
               n.iter=100000,n.chains=2,n.burnin=10000,n.thin=5)
ej3b.sim<-jags(data,inits,parameters,model.file="Ej3b.txt",
              n.iter=10000,n.chains=2,n.burnin=1000,n.thin=1)

#-Monitoring chain-

#Traza de la cadena
traceplot(ej3.sim)

#Cadena

#OpenBUGS
out<-ej3.sim$sims.list
out<-ej3b.sim$sims.list

#JAGS
out<-ej3.sim$BUGSoutput$sims.list
out<-ej3b.sim$BUGSoutput$sims.list

z<-out$beta[,2]
par(mfrow=c(2,2))
plot(z,type="l")
plot(cumsum(z)/(1:length(z)),type="l")
hist(z,freq=FALSE)
acf(z)

z<-out$beta
par(mfrow=c(1,1))
plot(z)

z<-out$beta
pairs(z)

#Resumen (estimadores)
#OpenBUGS
out.sum<-ej3.sim$summary
out.sum<-ej3b.sim$summary

#JAGS
out.sum<-ej3.sim$BUGSoutput$summary

#Tabla resumen
out.sum.t<-out.sum[grep("beta",rownames(out.sum)),c(1,3,7)]
out.sum.t<-cbind(out.sum.t,apply(out$beta,2,prob))
dimnames(out.sum.t)[[2]][4]<-"prob"
print(out.sum.t)

#DIC
out.dic<-ej3b.sim$DIC
out.dic<-ej3.sim$BUGSoutput$DIC
print(out.dic)

#Predictions
out.yf<-out.sum[grep("yf",rownames(out.sum)),]
or<-order(calif$MO)
ymin<-min(calif$SP,out.yf[,c(1,3,7)])
ymax<-max(calif$SP,out.yf[,c(1,3,7)])
par(mfrow=c(1,1))
plot(calif$MO,calif$SP,ylim=c(ymin,ymax))
lines(calif$MO[or],out.yf[or,1],lwd=2,col=2)
lines(calif$MO[or],out.yf[or,3],lty=2,col=2)
lines(calif$MO[or],out.yf[or,7],lty=2,col=2)

plot(calif$SP,out.yf[,1])
R2<-(cor(calif$SP,out.yf[,1]))^2
print(R2)


#--- Ejemplo 3.5---
#-Lectura de datos-
precio<-read.csv("http://allman.rhon.itam.mx/~lnieto/index_archivos/precio.csv")
anticipacion<-read.csv("http://allman.rhon.itam.mx/~lnieto/index_archivos/anticipacion.csv")
precio[,-1]<-as.matrix(precio[,-1])
anticipacion[,-1]<-as.matrix(anticipacion[,-1])
f<-function(x){ifelse(x==-1,NA,x)}
anticipacion<-sapply(anticipacion,f)
n<-dim(precio)[1]
m<-precio$num
y<-as.matrix(precio[,2:11])
x<-as.matrix(anticipacion[,2:11])
dia<-precio$dia
mes<-precio$mes
puente<-precio$puente+1
ano<-precio$ano-2016
pp<-m
ap<-m
for (i in 1:n){
  pp[i]<-mean(y[i,1:m[i]])
  ap[i]<-mean(x[i,1:m[i]])
}

#-Graficas-

#series de tiempo de precios
y1<-y[mes==1 & ano==1,]
ymin<-min(y1,na.rm=TRUE)
ymax<-max(y1,na.rm=TRUE)
xmax<-max(m[1:10])
plot(1:m[1],y1[1,1:m[1]],type="l",xlim=c(1,xmax),ylim=c(ymin,ymax),xlab="reserva",ylab="precio")
for (i in 2:10){
  lines(1:m[i],y1[i,1:m[i]],col=i)
}
title("Enero de 2017")
#series de tiempo de precios (anticipacion)
y1<-y[mes==1 & ano==1,]
x1<-x[mes==1 & ano==1,]
ymin<-min(y1,na.rm=TRUE)
ymax<-max(y1,na.rm=TRUE)
xmax<-max(x1[1:10,],na.rm=TRUE)
plot(x1[1,1:m[1]],y1[1,1:m[1]],xlim=c(0,xmax),ylim=c(ymin,ymax),xlab="anticipacion",ylab="precio",pch=19)
for (i in 2:10){
  points(x1[i,1:m[i]],y1[i,1:m[i]],col=i,pch=19)
}
title("Enero de 2017")
#anticipacion vs precio
plot(x,y,xlab="anticipacion",ylab="precio")
#boxplots de pp (precio promedio)
boxplot(pp~dia,main="precio prom. por dia")
boxplot(pp~mes,main="precio prom. por mes")
boxplot(pp~ano,main="precio prom. por ano")
boxplot(pp~puente,main="precio prom. por tipo")

#-Definiendo datos-
data<-list("n"=n,"y"=y/1000,"x"=x,"m"=m,"dia"=dia,"mes"=mes,"puente"=puente,"ano"=ano)

#-Definiendo inits-
inits<-function(){list(alpha=0,nu=0,beta=rep(0,n),gama=rep(0,7),delta=rep(0,12),epsilon=rep(0,3),theta=rep(0,2),tau=rep(1,n))}

#-Seleccionando parametros a monitorear-
parameters<-c("alpha.est","nu","beta.est","gama.est","delta.est","epsilon.est","theta.est","tau")

#-Corrida de codigo-
#OpenBUGS
mod1.sim<-bugs(data,inits,parameters,model.file="Ej35.txt",
               n.iter=1000,n.chains=2,n.burnin=100)
#JAGS
mod1.sim<-jags(data,inits,parameters,model.file="Ej35.txt",
              n.iter=1000,n.chains=2,n.burnin=100,n.thin=1)

#-Monitoreando la cadena-

#Traza de la cadena
#traceplot(mod1.sim)

#Cadena

#OpenBUGS
out<-mod1.sim$sims.list
#JAGS
out<-mod1.sim$BUGSoutput$sims.list
#
z<-out$alpha.est
par(mfrow=c(2,2))
plot(z,type="l")
plot(cumsum(z)/(1:length(z)),type="l")
hist(z,freq=FALSE)
acf(z)

#Resumen (estimadores)
#OpenBUGS
out.sum<-mod1.sim$summary
#JAGS
out.sum<-mod1.sim$BUGSoutput$summary
#
head(out.sum[,c(1,3,7)])
write.csv(head(out.sum[,c(1,3,7)]),file="pestim.csv")

par(mfrow=c(1,1))

#nu
out.nu<-out.sum[grep("nu",rownames(out.sum)),]
out.est<-out.nu
k<-1
print(out.est[c(1,3,7)])
ymin<-min(out.est[c(1,3,7)])
ymax<-max(out.est[c(1,3,7)])
plot(1:k,out.est[1],xlab="index",ylab="",ylim=c(ymin,ymax))
segments(1:k,out.est[3],1:k,out.est[7])
abline(h=0,col="grey70")
title("Precio: efecto anticipacion")

#beta
out.beta<-out.sum[grep("beta",rownames(out.sum)),]
out.est<-out.beta
k<-n
print(out.est[,c(1,3,7)])
ymin<-min(out.est[,c(1,3,7)])
ymax<-max(out.est[,c(1,3,7)])
plot(1:k,out.est[,1],xlab="index",ylab="",ylim=c(ymin,ymax))
segments(1:k,out.est[,3],1:k,out.est[,7])
abline(h=0,col="grey70")
title("Precio: efecto diario")

#gama
out.gama<-out.sum[grep("gama",rownames(out.sum)),]
out.est<-out.gama
k<-7
print(out.est[,c(1,3,7)])
ymin<-min(out.est[,c(1,3,7)])
ymax<-max(out.est[,c(1,3,7)])
plot(1:k,out.est[,1],xlab="dia",ylab="",ylim=c(ymin,ymax))
segments(1:k,out.est[,3],1:k,out.est[,7])
abline(h=0,col="grey70")
title("Precio: efecto dia de la semana")

#delta
out.delta<-out.sum[grep("delta",rownames(out.sum)),]
out.est<-out.delta
k<-12
print(out.est[,c(1,3,7)])
ymin<-min(out.est[,c(1,3,7)])
ymax<-max(out.est[,c(1,3,7)])
plot(1:k,out.est[,1],xlab="mes",ylab="",ylim=c(ymin,ymax))
segments(1:k,out.est[,3],1:k,out.est[,7])
abline(h=0,col="grey70")
title("Precio: efecto mensual")

#epsilon
out.epsilon<-out.sum[grep("epsilon",rownames(out.sum)),]
out.est<-out.epsilon
k<-3
print(out.est[,c(1,3,7)])
ymin<-min(out.est[,c(1,3,7)])
ymax<-max(out.est[,c(1,3,7)])
plot((1:k)+2016,out.est[,1],xlab="ano",ylab="",ylim=c(ymin,ymax))
segments((1:k)+2016,out.est[,3],(1:k)+2016,out.est[,7])
abline(h=0,col="grey70")
title("Precio: efecto anual")

#theta
out.theta<-out.sum[grep("theta",rownames(out.sum)),]
out.est<-out.theta
k<-2
print(out.est[,c(1,3,7)])
ymin<-min(out.est[,c(1,3,7)])
ymax<-max(out.est[,c(1,3,7)])
plot(1:k,out.est[,1],xlab="puente (no - si)",ylab="",ylim=c(ymin,ymax))
segments(1:k,out.est[,3],1:k,out.est[,7])
abline(h=0,col="grey70")
title("Precio: efecto puente")

#tau
out.tau<-out.sum[grep("tau",rownames(out.sum)),]
out.est<-out.tau
k<-n
print(out.est[,c(1,3,7)])
ymin<-min(out.est[,c(1,3,7)])
ymax<-max(out.est[,c(1,3,7)])
plot(1:k,out.est[,1],xlab="index- tau",ylab="",ylim=c(ymin,ymax))
segments(1:k,out.est[,3],1:k,out.est[,7])
abline(h=0,col="grey70")
title("Precio: precisiones diarias")

#DIC
#OpenBUGS
out.dic<-mod1.sim$DIC
#JAGS
out.dic<-mod1.sim$BUGSoutput$DIC
#
print(out.dic)

#Predicciones
#out.yf<-out.sum[grep("yf",rownames(out.sum)),]


#--- Ejemplo 4 ---
#TAREA


#--- Ejemplo 5 ---
#-Reading data-
mortality<-read.table("http://allman.rhon.itam.mx/~lnieto/index_archivos/mortality.txt",header=TRUE)
n<-nrow(mortality)
plot(mortality)
plot(mortality$x,mortality$y/mortality$n)
m<-1
nef<-c(100)
xf<-c(200)

#-Defining data-
data<-list("n"=n,"ne"=mortality$n,"y"=mortality$y,"x"=mortality$x,"m"=m,"nef"=nef,"xf"=xf)
data2<-list("n"=n,"y"=mortality$y/mortality$n,"x"=mortality$x,"m"=m,"xf"=xf)

#-Defining inits-
inits<-function(){list(beta=rep(0,2),yf1=rep(1,n),yf2=1)}
inits2<-function(){list(beta=rep(0,2),phy=1,yf1=rep(1,n),yf2=1)}

#-Selecting parameters to monitor-
parsa<-c("beta","lambda","yf1","yf2")
parsbc<-c("beta","p","yf1","yf2")
parsd<-c("beta","phy","yf1","yf2")

#-Running code-
#OpenBUGS
ej5a.sim<-bugs(data,inits,parsa,model.file="Ej5a.txt",
              n.iter=50000,n.chains=2,n.burnin=5000)
ej5b.sim<-bugs(data,inits,parsbc,model.file="Ej5b.txt",
               n.iter=50000,n.chains=2,n.burnin=5000)
ej5c.sim<-bugs(data,inits,parsbc,model.file="Ej5c.txt",
               n.iter=50000,n.chains=2,n.burnin=5000)
ej5d.sim<-bugs(data2,inits2,parsd,model.file="Ej5d.txt",
               n.iter=50000,n.chains=2,n.burnin=5000)
#JAGS
ej5a.sim<-jags(data,inits,parsa,model.file="Ej5a.txt",
              n.iter=50000,n.chains=2,n.burnin=5000,n.thin=1)
ej5b.sim<-jags(data,inits,parsbc,model.file="Ej5b.txt",
               n.iter=50000,n.chains=2,n.burnin=5000,n.thin=1)
ej5c.sim<-jags(data,inits,parasbc,model.file="Ej5c.txt",
               n.iter=50000,n.chains=2,n.burnin=5000,n.thin=1)
ej5d.sim<-jags(data2,inits2,parsd,model.file="Ej5d.txt",
               n.iter=50000,n.chains=2,n.burnin=5000,n.thin=1)

#-Monitoring chain-
ej5.sim<-ej5a.sim

#Traza de la cadena
traceplot(ej5.sim)

#Cadena

#OpenBUGS
out<-ej5.sim$sims.list

#JAGS
out<-ej5.sim$BUGSoutput$sims.list

z<-out$beta[,2]
par(mfrow=c(2,2))
plot(z,type="l")
plot(cumsum(z)/(1:length(z)),type="l")
hist(z,freq=FALSE)
acf(z)

z<-out$beta
par(mfrow=c(1,1))
plot(z)

#Resumen (estimadores)
#OpenBUGS
out.sum<-ej5.sim$summary

#JAGS
out.sum<-ej5.sim$BUGSoutput$summary

#Tabla resumen
out.sum.t<-out.sum[grep("beta",rownames(out.sum)),c(1,3,7)]
out.sum.t<-cbind(out.sum.t,apply(out$beta,2,prob))
dimnames(out.sum.t)[[2]][4]<-"prob"
print(out.sum.t)

#DIC
out.dic<-ej5c.sim$DIC
out.dic<-ej5.sim$BUGSoutput$DIC
print(out.dic)

#Predictions
out.yf<-out.sum[grep("yf1",rownames(out.sum)),]
or<-order(mortality$x)
ymin<-min(mortality$y,out.yf[,c(1,3,7)])
ymax<-max(mortality$y,out.yf[,c(1,3,7)])

par(mfrow=c(1,1))
plot(mortality$x,mortality$y,ylim=c(ymin,ymax))
#Modelo 1
lines(mortality$x[or],out.yf[or,1],lwd=2,col=2)
lines(mortality$x[or],out.yf[or,3],lty=2,col=2)
lines(mortality$x[or],out.yf[or,7],lty=2,col=2)
#Modelo 2
lines(mortality$x[or],out.yf[or,1],lwd=2,col=3)
lines(mortality$x[or],out.yf[or,3],lty=2,col=3)
lines(mortality$x[or],out.yf[or,7],lty=2,col=3)
#Modelo 3
lines(mortality$x[or],out.yf[or,1],lwd=2,col=4)
lines(mortality$x[or],out.yf[or,3],lty=2,col=4)
lines(mortality$x[or],out.yf[or,7],lty=2,col=4)
#Modelo 4
lines(mortality$x[or],out.yf[or,1],lwd=2,col=5)
lines(mortality$x[or],out.yf[or,3],lty=2,col=5)
lines(mortality$x[or],out.yf[or,7],lty=2,col=5)
#Modelo 5
lines(mortality$x[or],out.yf[or,1],lwd=2,col=6)
lines(mortality$x[or],out.yf[or,3],lty=2,col=6)
lines(mortality$x[or],out.yf[or,7],lty=2,col=6)

plot(mortality$y,out.yf[,1])
abline(a=0,b=1)
cor(mortality$y,out.yf[,1])

#Estimacion de tasas
out.tasa<-out.sum[grep("lambda",rownames(out.sum)),]
out.tasa<-out.sum[grep("p",rownames(out.sum)),]
or<-order(mortality$x)
ymin<-min(mortality$y/mortality$n,out.tasa[,c(1,3,7)])
ymax<-max(mortality$y/mortality$n,out.tasa[,c(1,3,7)])

par(mfrow=c(1,1))
plot(mortality$x,mortality$y/mortality$n,ylim=c(ymin,ymax))
#Modelo 1
lines(mortality$x[or],out.tasa[or,1],lwd=2,col=2)
lines(mortality$x[or],out.tasa[or,3],lty=2,col=2)
lines(mortality$x[or],out.tasa[or,7],lty=2,col=2)
#Modelo 2
lines(mortality$x[or],out.tasa[or,1],lwd=2,col=3)
lines(mortality$x[or],out.tasa[or,3],lty=2,col=3)
lines(mortality$x[or],out.tasa[or,7],lty=2,col=3)
#Modelo 3
lines(mortality$x[or],out.tasa[or,1],lwd=2,col=4)
lines(mortality$x[or],out.tasa[or,3],lty=2,col=4)
lines(mortality$x[or],out.tasa[or,7],lty=2,col=4)
#Modelo 4
lines(mortality$x[or],out.tasa[or,1],lwd=2,col=5)
lines(mortality$x[or],out.tasa[or,3],lty=2,col=5)
lines(mortality$x[or],out.tasa[or,7],lty=2,col=5)
#Modelo 5
lines(mortality$x[or],out.tasa[or,1],lwd=2,col=6)
lines(mortality$x[or],out.tasa[or,3],lty=2,col=6)
lines(mortality$x[or],out.tasa[or,7],lty=2,col=6)


#--- Ejemplo 6 ---
#-Reading data-
desastres<-read.table("http://allman.rhon.itam.mx/~lnieto/index_archivos/desastres.txt",header=TRUE)
n<-nrow(desastres)
plot(desastres,type="l")
plot(desastres[2:n,2]-desastres[1:(n-1),2],type="l")

#-Defining data-
data<-list("n"=n,"y"=desastres$No.Desastres,"x"=desastres$Anho)
data2<-list("n"=n,"y"=c(desastres$No.Desastres[1:(n-6)],rep(NA,6)),"x"=desastres$Anho)
data3<-list("n"=n,"y"=desastres$No.Desastres+0.1,"x"=desastres$Anho/1000)

#-Defining inits-
initsa1<-function(){list(beta=rep(0,2),yf1=rep(1,n))}
initsa2<-function(){list(beta=rep(0,2),aux=1,yf1=rep(1,n))}
initsa3<-function(){list(beta=rep(0,2),a=1,yf1=rep(1,n))}
initsb1<-function(){list(beta=rep(0,2),aux2=1,yf1=rep(1,n))}
initsb2<-function(){list(beta=rep(0,2),aux=1,aux2=1,yf1=rep(1,n))}
initsb3<-function(){list(beta=rep(0,2),a=1,aux2=1,yf1=rep(1,n))}
initsc<-function(){list(beta=rep(0,n),tau.b=1,yf1=rep(1,n))}
initsd<-function(){list(mu=rep(1,n),tau.b=1,yf1=rep(1,n))}

#-Selecting parameters to monitor-
parsa1<-c("beta","yf1","mu")
parsa2<-c("beta","yf1","mu","r")
parsa3<-c("beta","yf1","mu","a")
parsb1<-c("beta","yf1","mu","tau")
parsb2<-c("beta","yf1","mu","tau","r")
parsb3<-c("beta","yf1","mu","tau","a")
parscd<-c("tau.b","yf1","mu")

#-Running code-
#OpenBUGS
ej6a1.sim<-bugs(data,initsa1,parsa1,model.file="Ej6a1.txt",
               n.iter=10000,n.chains=2,n.burnin=1000,n.thin=2)
ej6a2.sim<-bugs(data,initsa2,parsa2,model.file="Ej6a2.txt",
                n.iter=10000,n.chains=2,n.burnin=1000,n.thin=2)
ej6a3.sim<-bugs(data3,initsa3,parsa3,model.file="Ej6a3.txt",
                n.iter=10000,n.chains=2,n.burnin=1000,n.thin=2)
ej6b1.sim<-bugs(data,initsb1,parsb1,model.file="Ej6b1.txt",
               n.iter=10000,n.chains=2,n.burnin=1000,n.thin=2)
ej6b2.sim<-bugs(data,initsb2,parsb2,model.file="Ej6b2.txt",
                n.iter=10000,n.chains=2,n.burnin=1000,n.thin=2)
ej6b3.sim<-bugs(data3,initsb3,parsb3,model.file="Ej6b3.txt",
                n.iter=10000,n.chains=2,n.burnin=1000,n.thin=2)
ej6c.sim<-bugs(data,initsc,parscd,model.file="Ej6c.txt",
               n.iter=10000,n.chains=2,n.burnin=1000,n.thin=2)
ej6d.sim<-bugs(data,initsd,parscd,model.file="Ej6d.txt",
               n.iter=10000,n.chains=2,n.burnin=1000,n.thin=2)
#JAGS
ej6a1.sim<-jags(data,initsa1,parsa1,model.file="Ej6a1.txt",
                n.iter=10000,n.chains=2,n.burnin=1000,n.thin=2)
ej6a2.sim<-jags(data,initsa2,parsa2,model.file="Ej6a2.txt",
                n.iter=10000,n.chains=2,n.burnin=1000,n.thin=2)
ej6a3.sim<-jags(data3,initsa3,parsa3,model.file="Ej6a3.txt",
                n.iter=10000,n.chains=2,n.burnin=1000,n.thin=2)
ej6b1.sim<-jags(data,initsb1,parsb1,model.file="Ej6b1.txt",
                n.iter=10000,n.chains=2,n.burnin=1000,n.thin=2)
ej6b2.sim<-jags(data,initsb2,parsb2,model.file="Ej6b2.txt",
                n.iter=10000,n.chains=2,n.burnin=1000,n.thin=2)
ej6b3.sim<-jags(data3,initsb3,parsb3,model.file="Ej6b3.txt",
                n.iter=10000,n.chains=2,n.burnin=1000,n.thin=2)
ej6c.sim<-jags(data,initsc,parscd,model.file="Ej6c.txt",
               n.iter=10000,n.chains=2,n.burnin=1000,n.thin=2)
ej6d.sim<-jags(data,initsd,parscd,model.file="Ej6d.txt",
               n.iter=10000,n.chains=2,n.burnin=1000,n.thin=2)

#-Monitoring chain-
ej6.sim<-ej6a1.sim
  
#Traza de la cadena
traceplot(ej6.sim)

#Cadena

#OpenBUGS
out<-ej6.sim$sims.list

#JAGS
out<-ej6.sim$BUGSoutput$sims.list

z<-out$mu[,1]
par(mfrow=c(2,2))
plot(z,type="l")
plot(cumsum(z)/(1:length(z)),type="l")
hist(z,freq=FALSE)
acf(z)

z<-out$beta
par(mfrow=c(1,1))
plot(z)

#Resumen (estimadores)
#OpenBUGS
out.sum<-ej6.sim$summary

#JAGS
out.sum<-ej6.sim$BUGSoutput$summary

#Tabla resumen
out.sum.t<-out.sum[grep("beta",rownames(out.sum)),c(1,3,7)]
out.sum.t<-cbind(out.sum.t,apply(out$beta,2,prob))
dimnames(out.sum.t)[[2]][4]<-"prob"
print(out.sum.t)

#DIC
out.dic<-ej6.sim$DIC
out.dic<-ej6.sim$BUGSoutput$DIC
print(out.dic)

#Predictions
out.yf<-out.sum[grep("yf1",rownames(out.sum)),]
ymin<-min(desastres[,2],out.yf[,c(1,3,7)])
ymax<-max(desastres[,2],out.yf[,c(1,3,7)])

par(mfrow=c(1,1))
plot(desastres,type="l",col="grey80",ylim=c(ymin,ymax))
lines(desastres[,1],out.yf[,1],lwd=2,col=2)
lines(desastres[,1],out.yf[,3],lty=2,col=2)
lines(desastres[,1],out.yf[,7],lty=2,col=2)
lines(desastres[,1],out.yf[,5],lwd=2,col=4)

#Medias
out.mu<-out.sum[grep("mu",rownames(out.sum)),]
par(mfrow=c(1,1))
plot(desastres,type="l",col="grey80")
lines(desastres[,1],out.mu[,1],lwd=2,col=2)
lines(desastres[,1],out.mu[,3],lty=2,col=2)
lines(desastres[,1],out.mu[,7],lty=2,col=2)


#--- Ejemplo 7 ---
#-Reading data-
leucemia<-read.table("http://allman.rhon.itam.mx/~lnieto/index_archivos/leucemia.txt",header=TRUE)
n<-nrow(leucemia)
par(mfrow=c(2,2))
plot(leucemia$Obs)
plot(leucemia$Obs/leucemia$Pops*10000)
plot(leucemia$Obs/leucemia$Esp)
abline(h=1,col=2)

#-Defining data-
data<-list("n"=n,"y"=leucemia$Obs,"ne"=leucemia$Pops/10000)
data2<-list("n"=n,"y"=leucemia$Obs,"ne"=leucemia$Pops/10000,"C"=leucemia$Cancer,"P"=leucemia$Place,"A"=leucemia$Age)

#-Defining inits-
initsa<-function(){list(theta=1,yf1=rep(1,n))}
initsb<-function(){list(theta=rep(1,n),yf1=rep(1,n))}
initsc<-function(){list(theta=rep(1,n),a=1,b=1,yf1=rep(1,n))}
initsd<-function(){list(alpha=0,beta=rep(0,2),gama=rep(0,2),delta=rep(0,2),yf1=rep(1,n))}

#-Selecting parameters to monitor-
parsa<-c("theta","yf1")
parsc<-c("theta","eta","yf1")
parsd<-c("alpha.adj","beta.adj","gama.adj","delta.adj","yf1")

#-Running code-
#OpenBUGS
ej7a.sim<-bugs(data,initsa,parsa,model.file="Ej7a.txt",
               n.iter=50000,n.chains=2,n.burnin=5000)
ej7b.sim<-bugs(data,initsb,parsa,model.file="Ej7b.txt",
               n.iter=50000,n.chains=2,n.burnin=5000)
ej7c.sim<-bugs(data,initsc,parsc,model.file="Ej7c.txt",
               n.iter=100000,n.chains=2,n.burnin=10000)
ej7d.sim<-bugs(data2,initsd,parsd,model.file="Ej7d.txt",
               n.iter=50000,n.chains=2,n.burnin=5000)

#JAGS
ej7a.sim<-jags(data,initsa,parsa,model.file="Ej7a.txt",
               n.iter=50000,n.chains=2,n.burnin=5000,n.thin=1)
ej7b.sim<-jags(data,initsb,parsa,model.file="Ej7b.txt",
               n.iter=50000,n.chains=2,n.burnin=5000,n.thin=1)
ej7c.sim<-jags(data,initsc,parsc,model.file="Ej7c.txt",
               n.iter=100000,n.chains=2,n.burnin=10000,n.thin=1)
ej7d.sim<-jags(data2,initsd,parsd,model.file="Ej7d.txt",
               n.iter=50000,n.chains=2,n.burnin=5000,n.thin=1)

#-Monitoring chain-
ej7.sim<-ej7a.sim

#Traza de la cadena
traceplot(ej7.sim)

#Cadena

#OpenBUGS
outa<-ej7a.sim$sims.list
outb<-ej7b.sim$sims.list
outc<-ej7c.sim$sims.list
outd<-ej7d.sim$sims.list

#JAGS
outa<-ej7a.sim$BUGSoutput$sims.list
outb<-ej7b.sim$BUGSoutput$sims.list
outc<-ej7c.sim$BUGSoutput$sims.list
outc<-ej7d.sim$BUGSoutput$sims.list

z<-outa$theta
par(mfrow=c(2,2))
plot(z,type="l")
plot(cumsum(z)/(1:length(z)),type="l")
hist(z,freq=FALSE)
acf(z)

#Resumen (estimadores)
#OpenBUGS
outa.sum<-ej7a.sim$summary
outb.sum<-ej7b.sim$summary
outc.sum<-ej7c.sim$summary
outd.sum<-ej7d.sim$summary

#JAGS
outa.sum<-ej7a.sim$BUGSoutput$summary
outb.sum<-ej7b.sim$BUGSoutput$summary
outc.sum<-ej7c.sim$BUGSoutput$summary
outd.sum<-ej7d.sim$BUGSoutput$summary

#Tabla resumen
out<-outb
out.sum<-outb.sum
out.sum.t<-out.sum[grep("theta",rownames(out.sum)),c(1,3,7)]
out.sum.t<-cbind(out.sum.t,apply(out$theta,2,prob))
dimnames(out.sum.t)[[2]][4]<-"prob"
print(out.sum.t)

#DIC
#OpenBUGS
outa.dic<-ej7a.sim$DIC
outb.dic<-ej7b.sim$DIC
outc.dic<-ej7c.sim$DIC
outd.dic<-ej7d.sim$DIC

#JAGS
outa.dic<-ej7a.sim$BUGSoutput$DIC
outb.dic<-ej7b.sim$BUGSoutput$DIC
outc.dic<-ej7c.sim$BUGSoutput$DIC
outd.dic<-ej7d.sim$BUGSoutput$DIC

print(outa.dic)
print(outb.dic)
print(outc.dic)
print(outd.dic)

#Estimaciones
outa.p<-outa.sum[grep("theta",rownames(outa.sum)),]
outb.p<-outb.sum[grep("theta",rownames(outb.sum)),]
outc.p<-outc.sum[grep("theta",rownames(outc.sum)),]
outc.eta<-outc.sum[grep("eta",rownames(outc.sum)),]

#x vs. y
xmin<-0
xmax<-10
ymin<-0
ymax<-5
par(mfrow=c(1,1))
plot(leucemia$Obs/leucemia$Pops*10000,type="p",col="grey50",xlim=c(xmin,xmax),ylim=c(ymin,ymax))
#
out.p<-outb.p
points(out.p[,1],col=2,pch=16,cex=0.5)
segments(1:8,out.p[,3],1:8,out.p[,7],col=2)
#
out.p<-outc.p
points((1:8)+0.2,out.p[,1],col=4,pch=16,cex=0.5)
segments((1:8)+0.2,out.p[,3],(1:8)+0.2,out.p[,7],col=4)
#
points(xmax-0.2,sum(leucemia$Obs)/sum(leucemia$Pops)*10000)
#
out.p<-outa.p
points(xmax-0.2,out.p[1],col=3,pch=16,cex=0.5)
segments(xmax-0.2,out.p[3],xmax-0.2,out.p[7],col=3)
#
out.p<-outc.eta
points(xmax,out.p[1],col=4,pch=16,cex=0.5)
segments(xmax,out.p[3],xmax,out.p[7],col=4)


#--- Ejemplo 8 ---
#-Reading data-
reclama<-read.table("http://allman.rhon.itam.mx/~lnieto/index_archivos/reclama.txt",header=TRUE)
n<-nrow(reclama)
par(mfrow=c(2,2))
plot(reclama$r)
plot(reclama$n,ylim=c(0,max(reclama$n)))
plot(reclama$r/reclama$n)

#-Defining data-
data<-list("n"=n,"y"=reclama$r,"ne"=reclama$n)

#-Defining inits-
initsa<-function(){list(p=0.5,yf1=rep(1,n))}
initsb<-function(){list(p=rep(0.5,n),yf1=rep(1,n))}
initsc<-function(){list(p=rep(0.5,n),a=1,b=1,yf1=rep(1,n))}

#-Selecting parameters to monitor-
parsa<-c("p","yf1")
parsc<-c("p","eta","yf1")

#-Running code-
#OpenBUGS
ej8a.sim<-bugs(data,initsa,parsa,model.file="Ej8a.txt",
               n.iter=50000,n.chains=2,n.burnin=5000)
ej8b.sim<-bugs(data,initsb,parsa,model.file="Ej8b.txt",
               n.iter=50000,n.chains=2,n.burnin=5000)
ej8c.sim<-bugs(data,initsc,parsc,model.file="Ej8c.txt",
               n.iter=100000,n.chains=2,n.burnin=10000)

#JAGS
ej8a.sim<-jags(data,initsa,parsa,model.file="Ej8a.txt",
               n.iter=50000,n.chains=2,n.burnin=5000,n.thin=1)
ej8b.sim<-jags(data,initsb,parsa,model.file="Ej8b.txt",
               n.iter=50000,n.chains=2,n.burnin=5000,n.thin=1)
ej8c.sim<-jags(data,initsc,parsc,model.file="Ej8c.txt",
               n.iter=100000,n.chains=2,n.burnin=10000,n.thin=1)

#-Monitoring chain-
el8.sim<-ej8a.sim

#Traza de la cadena
traceplot(ej8.sim)

#Cadena

#OpenBUGS
outa<-ej8a.sim$sims.list
outb<-ej8b.sim$sims.list
outc<-ej8c.sim$sims.list

#JAGS
outa<-ej8a.sim$BUGSoutput$sims.list
outb<-ej8b.sim$BUGSoutput$sims.list
outc<-ej8c.sim$BUGSoutput$sims.list

z<-outb$p[,2]
par(mfrow=c(2,2))
plot(z,type="l")
plot(cumsum(z)/(1:length(z)),type="l")
hist(z,freq=FALSE)
acf(z)

#Resumen (estimadores)
#OpenBUGS
outa.sum<-ej8a.sim$summary
outb.sum<-ej8b.sim$summary
outc.sum<-ej8c.sim$summary

#JAGS
outa.sum<-ej8a.sim$BUGSoutput$summary
outb.sum<-ej8b.sim$BUGSoutput$summary
outc.sum<-ej8c.sim$BUGSoutput$summary

#Tabla resumen
out.sum.t<-out.sum[grep("p",rownames(out.sum)),c(1,3,7)]
out.sum.t<-cbind(out.sum.t,apply(out$p,2,prob))
dimnames(out.sum.t)[[2]][4]<-"prob"
print(out.sum.t)

#DIC
#OpenBUGS
outa.dic<-ej8a.sim$DIC
outb.dic<-ej8b.sim$DIC
outc.dic<-ej8c.sim$DIC

#JAGS
outa.dic<-ej8a.sim$BUGSoutput$DIC
outb.dic<-ej8b.sim$BUGSoutput$DIC
outc.dic<-ej8c.sim$BUGSoutput$DIC

print(outa.dic)
print(outb.dic)
print(outc.dic)

#Estimaciones
outa.p<-outa.sum[grep("p",rownames(outa.sum)),]
outb.p<-outb.sum[grep("p",rownames(outb.sum)),]
outc.p<-outc.sum[grep("p",rownames(outc.sum)),]
outc.eta<-outc.sum[grep("eta",rownames(outc.sum)),]

#x vs. y
xmin<-0
xmax<-12
ymin<-0
ymax<-1
par(mfrow=c(1,1))
plot(reclama$r/reclama$n,type="p",col="grey50",xlim=c(xmin,xmax),ylim=c(ymin,ymax))
#
out.p<-outb.p
points(out.p[,1],col=2,pch=16,cex=0.5)
segments(1:10,out.p[,3],1:10,out.p[,7],col=2)
#
out.p<-outc.p
points((1:10)+0.2,out.p[,1],col=4,pch=16,cex=0.5)
segments((1:10)+0.2,out.p[,3],(1:10)+0.2,out.p[,7],col=4)
#
points(xmax-0.2,sum(reclama$r)/sum(reclama$n))
#
out.p<-outa.p
points(xmax-0.2,out.p[1],col=3,pch=16,cex=0.5)
segments(xmax-0.2,out.p[3],xmax-0.2,out.p[7],col=3)
#
out.p<-outc.eta
points(xmax,out.p[1],col=4,pch=16,cex=0.5)
segments(xmax,out.p[3],xmax,out.p[7],col=4)


#--- Ejemplo 9 ---
#-Reading data-
milk<-read.table("http://allman.rhon.itam.mx/~lnieto/index_archivos/milk.txt",header=TRUE)
milk$t<-1970:1982
n<-nrow(milk)
plot(milk$x,milk$y,type="n")
text(milk$x,milk$y,labels=milk$t,cex=0.5,col=1)
plot(milk$t,milk$y,type="l")
plot(milk$t,milk$x,type="l")

#-Defining data-
m<-2
data<-list("n"=n,"m"=m,"y"=milk$y,"x"=milk$x,"t"=milk$t)
data<-list("n"=n,"m"=m,"y"=milk$y/max(milk$y),"x"=milk$x/max(milk$x),"t"=milk$t)
data<-list("n"=n,"m"=m,"y"=c(milk$y/max(milk$y)[1:(n-2)],NA,NA),"x"=milk$x/max(milk$x),"t"=milk$t)

#-Defining inits-
initsa1<-function(){list(beta=rep(0,2),tau=1,yf1=rep(1,n))}
initsa2<-function(){list(beta=rep(0,5),tau=1,yf1=rep(1,n))}
initsb1<-function(){list(beta=rep(0,n+m),tau.y=1,tau.b=1,yf1=rep(0,n+m))}
initsb2<-function(){list(beta=rep(0,n+m),tau.y=1,yf1=rep(0,n+m))}
initsc<-function(){list(beta=rep(0,n+m),tau.y=1,tau.b=1,yf1=rep(0,n+m))}
initsd<-function(){list(alpha=rep(0,n+m),beta=rep(0,n+m),tau.y=1,tau.b=1,tau.a=1,yf1=rep(0,n+m))}

#-Selecting parameters to monitor-
parsa<-c("beta","tau","yf1")
parsb1<-c("beta","tau.y","tau.b","yf1")
parsb2<-c("beta","tau.y","tau.b","yf1","g")
parsc<-c("beta","tau.y","tau.b","yf1","g")
parsd<-c("alpha","beta","tau.y","tau.b","tau.a","yf1")

#-Running code-
#OpenBUGS
ej9a.sim<-bugs(data,initsa1,parsa,model.file="Ej9a.txt",
               n.iter=50000,n.chains=2,n.burnin=5000)
ej9b.sim<-bugs(data,initsb1,parsb1,model.file="Ej9b.txt",
               n.iter=50000,n.chains=2,n.burnin=5000)
ej9c.sim<-bugs(data,initsc,parsc,model.file="Ej9c.txt",
               n.iter=50000,n.chains=2,n.burnin=5000)
ej9d.sim<-bugs(data,initsd,parsd,model.file="Ej9d.txt",
               n.iter=50000,n.chains=2,n.burnin=5000)
#JAGS
ej9a.sim<-jags(data,initsa1,parsa,model.file="Ej9a.txt",
              n.iter=50000,n.chains=2,n.burnin=5000,n.thin=1)
ej9b.sim<-jags(data,initsb1,parsb1,model.file="Ej9b.txt",
               n.iter=50000,n.chains=2,n.burnin=5000,n.thin=1)
ej9c.sim<-jags(data,initsc,parsc,model.file="Ej9c.txt",
               n.iter=50000,n.chains=2,n.burnin=5000,n.thin=1)
ej9d.sim<-jags(data,initsd,parsd,model.file="Ej9d.txt",
               n.iter=50000,n.chains=2,n.burnin=5000,n.thin=1)

#-Monitoring chain-
ej9.sim<-ej9a.sim

#Traza de la cadena
traceplot(ej9.sim)

#Cadena

#OpenBUGS
out<-ej9.sim$sims.list

#JAGS
out<-ej9.sim$BUGSoutput$sims.list

z<-out$beta
par(mfrow=c(2,2))
plot(z,type="l")
plot(cumsum(z)/(1:length(z)),type="l")
hist(z,freq=FALSE)
acf(z)

#Resumen (estimadores)
#OpenBUGS
out.sum<-ej9.sim$summary

#JAGS
out.sum<-ej9.sim$BUGSoutput$summary

#Tabla resumen
out.sum.t<-out.sum[grep("beta",rownames(out.sum)),c(1,3,7)]
out.sum.t<-cbind(out.sum.t,apply(out$beta,2,prob))
dimnames(out.sum.t)[[2]][4]<-"prob"
print(out.sum.t)

#DIC
out.dic<-ej9.sim$DIC
out.dic<-ej9.sim$BUGSoutput$DIC
print(out.dic)

#Predictions
out.yf<-out.sum[grep("yf1",rownames(out.sum)),]
ymin<-min(data$y,out.yf[,c(1,3,7)])
ymax<-max(data$y,out.yf[,c(1,3,7)])
xmin<-min(data$t)
xmax<-max(data$t+m)

#x vs. y
par(mfrow=c(1,1))
plot(data$x,data$y,type="p",col="grey50",ylim=c(ymin,ymax))
points(data$x,out.yf[,1],col=2,pch=16,cex=0.5)
segments(data$x,out.yf[,3],data$x,out.yf[,7],col=2)

#t vs y
par(mfrow=c(1,1))
plot(data$t,data$y,type="b",col="grey80",ylim=c(ymin,ymax),xlim=c(xmin,xmax))
lines(data$t,out.yf[1:n,1],col=2)
lines(data$t,out.yf[1:n,3],col=2,lty=2)
lines(data$t,out.yf[1:n,7],col=2,lty=2)
lines(data$t[n]:(data$t[n]+m),out.yf[n:(n+m),1],col=4)
lines(data$t[n]:(data$t[n]+m),out.yf[n:(n+m),3],col=4,lty=2)
lines(data$t[n]:(data$t[n]+m),out.yf[n:(n+m),7],col=4,lty=2)

#betas
out.beta<-out.sum[grep("beta",rownames(out.sum)),]
ymin<-min(out.beta[,c(1,3,7)])
ymax<-max(out.beta[,c(1,3,7)])
plot(out.beta[,1],type="l",ylim=c(ymin,ymax))
lines(out.beta[,3],lty=2)
lines(out.beta[,7],lty=2)


#--- Ejemplo 10 ---
#-Reading data-
mercado<-read.table("http://allman.rhon.itam.mx/~lnieto/index_archivos/mercado.txt",header=TRUE)
mercado.ts<-ts(mercado,start=c(1990,1),end=c(1991,52),frequency=52)
n<-nrow(mercado)
mercado$Tiempo<-1:n
plot(mercado.ts)
pairs(mercado)
cor(mercado)

#-Defining data-
data<-list("n"=n,"y"=mercado$SHARE,"x1"=mercado$PRICE,"x2"=mercado$OPROM,"x3"=mercado$CPROM)
data<-list("n"=n,"y"=scale(mercado$SHARE)[1:n],"x1"=mercado$PRICE,"x2"=mercado$OPROM,"x3"=mercado$CPROM)
data<-list("n"=n,"y"=c(scale(mercado$SHARE)[1:(n-4)],NA,NA,NA,NA),"x1"=mercado$PRICE,"x2"=mercado$OPROM,"x3"=mercado$CPROM)

#-Defining inits-
initsa<-function(){list(alpha=0,beta=rep(0,3),tau=1,yf1=rep(1,n))}
initsb<-function(){list(alpha=rep(0,n),beta=matrix(0,nrow=3,ncol=n),tau=1,tau.a=1,tau.b=rep(1,3),yf1=rep(1,n))}
initsc<-function(){list(alpha=0,beta=matrix(0,nrow=3,ncol=n),tau=1,yf1=rep(1,n))}
initsd<-function(){list(beta=rep(0,n),tau=1,tau.b=1,yf1=rep(1,n))}

#-Selecting parameters to monitor-
parameters<-c("alpha","beta","tau","yf1")

#-Running code-
#OpenBUGS
ej10a.sim<-bugs(data,initsa,parameters,model.file="Ej10a.txt",
               n.iter=10000,n.chains=2,n.burnin=1000)
ej10b.sim<-bugs(data,initsb,parameters,model.file="Ej10b.txt",
               n.iter=10000,n.chains=2,n.burnin=1000)
ej10c.sim<-bugs(data,initsc,parameters,model.file="Ej10c.txt",
               n.iter=10000,n.chains=2,n.burnin=1000)
ej10d.sim<-bugs(data,initsd,parameters,model.file="Ej10d.txt",
               n.iter=10000,n.chains=2,n.burnin=1000)
#JAGS
ej10a.sim<-jags(data,initsa,parameters,model.file="Ej10a.txt",
               n.iter=10000,n.chains=2,n.burnin=1000,n.thin=1)
ej10b.sim<-jags(data,initsb,parameters,model.file="Ej10b.txt",
               n.iter=10000,n.chains=2,n.burnin=1000,n.thin=1)
ej10c.sim<-jags(data,initsc,parameters,model.file="Ej10c.txt",
               n.iter=10000,n.chains=2,n.burnin=1000,n.thin=1)
ej10d.sim<-jags(data,initsd,parameters,model.file="Ej10d.txt",
               n.iter=10000,n.chains=2,n.burnin=1000,n.thin=1)

#-Monitoring chain-
ej10.sim<-ej8a.sim

#OpenBUGS
out<-ej10.sim$sims.list

#JAGS
out<-ej10.sim$BUGSoutput$sims.list

z<-out$alpha
par(mfrow=c(2,2))
plot(z,type="l")
plot(cumsum(z)/(1:length(z)),type="l")
hist(z,freq=FALSE)
acf(z)

#Resumen (estimadores)
#OpenBUGS
out.sum<-ej10.sim$summary

#JAGS
out.sum<-ej10.sim$BUGSoutput$summary

#Tabla resumen
out.sum.t<-out.sum[grep("beta",rownames(out.sum)),c(1,3,7)]
out.sum.t<-cbind(out.sum.t,apply(out$beta,2,prob))
dimnames(out.sum.t)[[2]][4]<-"prob"
print(out.sum.t)

#DIC
out.dic<-ej10.sim$DIC
out.dic<-ej10.sim$BUGSoutput$DIC
print(out.dic)

#Predictions
out.yf<-out.sum[grep("yf1",rownames(out.sum)),]
y<-data$y
ymin<-min(y,out.yf[,c(1,3,7)])
ymax<-max(y,out.yf[,c(1,3,7)])

#x1 vs. y
x<-data$x1
par(mfrow=c(1,1))
plot(x,y,type="p",col="grey50",ylim=c(ymin,ymax))
points(x,out.yf[,1],col=2,pch=16,cex=0.5)
segments(x,out.yf[,3],x,out.yf[,7],col=2)
#x2 vs. y
x<-data$x2
par(mfrow=c(1,1))
plot(x,y,type="p",col="grey50",ylim=c(ymin,ymax))
points(x,out.yf[,1],col=2,pch=16,cex=0.5)
segments(x,out.yf[,3],x,out.yf[,7],col=2)
#x3 vs. y
x<-data$x3
par(mfrow=c(1,1))
plot(x,y,type="p",col="grey50",ylim=c(ymin,ymax))
points(x,out.yf[,1],col=2,pch=16,cex=0.5)
segments(x,out.yf[,3],x,out.yf[,7],col=2)
#t vs. y
x<-mercado$Tiempo
par(mfrow=c(1,1))
plot(x,y,type="p",col="grey50",ylim=c(ymin,ymax))
points(x,out.yf[,1],col=2,pch=16,cex=0.5)
segments(x,out.yf[,3],x,out.yf[,7],col=2)
par(mfrow=c(1,1))
plot(x,y,type="l",col="grey50",ylim=c(ymin,ymax))
lines(x,out.yf[,1],col=2,cex=0.5)
lines(x,out.yf[,3],col=2,lty=2)
lines(x,out.yf[,7],col=2,lty=2)

#betas
out.beta<-out.sum[grep("beta",rownames(out.sum)),]
plot(out.beta[1:104,1],type="l")
plot(out.beta[105:208,1],type="l")
plot(out.beta[209:312,1],type="l")

#alpha
out.alpha<-out.sum[grep("alpha",rownames(out.sum)),]
plot(out.alpha[,1],type="l")


#--- Ejemplo 11 ---
install.packages("SemiPar")
install.packages("maps")
install.packages("interp")
install.packages("gstat")
library(SemiPar)
library(maps)
library(interp)
library(gstat)
library(sp)

#-Loading data-
data(scallop)
n<-nrow(scallop)
pairs(scallop)
hist(scallop$tot.catch,freq=FALSE)
scallop$lgcatch<-log(scallop$tot.catch+1)
hist(scallop$lgcatch)

#Maps, contours and 3Dplots
map("usa",xlim=c(-74,-71),ylim=c(38.2,41.5))
points(scallop$longitude,scallop$latitude,cex=0.75,pch=20)
int.scp<-interp(scallop$longitude,scallop$latitude,scallop$lgcatch)
contour(int.scp,add=TRUE)
#
map("usa",xlim=c(-74,-71),ylim=c(38.2,41.5))
image(int.scp)
#
persp(int.scp)

#variograma
scallop2<-scallop
coordinates(scallop2)=~longitude+latitude
scallop.var<-variogram(lgcatch~longitude+latitude,scallop2)
plot(scallop.var)

#-Defining data-
y<-scallop$lgcatch
s1<-scallop$longitude
s2<-scallop$latitude
m<-2
s1f<-c(-71,-72.75)
s2f<-c(40,39.5)
data<-list("n"=n,"y"=y,"s1"=s1,"s2"=s2,"m"=m,"s1f"=s1f,"s2f"=s2f)

#-Defining inits-
inits<-function(){list(alpha=0,tau=1,w=rep(0,n),tau.w=1,phi=1,yf1=rep(0,n),wf=rep(0,m),yf2=rep(0,m))}

#-Selecting parameters to monitor-
parameters<-c("alpha","tau","w","tau.w","phi","yf1","yf2")

#-Running code-
#OpenBUGS
ej11.sim<-bugs(data,inits,parameters,model.file="Ej11.txt",
               n.iter=3000,n.chains=2,n.burnin=1000,n.thin=1)
#JAGS
ej11.sim<-jags(data,inits,parameters,model.file="Ej11.txt",
               n.iter=3000,n.chains=2,n.burnin=1000,n.thin=1)

#-Monitoring chain-

#Traza de la cadena
traceplot(ej11.sim)

#Cadena

#OpenBUGS
out<-ej11.sim$sims.list

#JAGS
out<-ej11.sim$BUGSoutput$sims.list

z<-out$alpha
par(mfrow=c(2,2))
plot(z,type="l")
plot(cumsum(z)/(1:length(z)),type="l")
hist(z,freq=FALSE)
acf(z)

#Resumen (estimadores)
#OpenBUGS
out.sum<-ej11.sim$summary

#JAGS
out.sum<-ej11.sim$BUGSoutput$summary

#Tabla resumen
out.w<-out.sum[grep("w",rownames(out.sum)),c(1,3,7)]
out.w<-out.w[-nrow(out.w),]
out.w<-cbind(out.w,apply(out$w,2,prob))
dimnames(out.w)[[2]][4]<-"prob"
print(out.w)

#w
out.w<-out.sum[grep("w",rownames(out.sum)),]
out.w<-out.w[-nrow(out.w),]
out.est<-out.w
k<-n
ymin<-min(out.est[,c(1,3,7)])
ymax<-max(out.est[,c(1,3,7)])
par(mfrow=c(1,1))
plot(1:k,out.est[,1],xlab="index",ylab="",ylim=c(ymin,ymax))
segments(1:k,out.est[,3],1:k,out.est[,7])
abline(h=0,col="grey70")
title("Efecto espacial")

#DIC
out.dic<-ej11.sim$DIC
out.dic<-ej11.sim$BUGSoutput$DIC
print(out.dic)

#Predictions
out.yf<-out.sum[grep("yf1",rownames(out.sum)),]
#longitude
or<-order(s1)
ymin<-min(y,out.yf[,c(1,3,7)])
ymax<-max(y,out.yf[,c(1,3,7)])
par(mfrow=c(1,1))
plot(s1,y,ylim=c(ymin,ymax),xlab="longitude")
points(s1[or],out.yf[or,1],pch=20,col=2)
segments(s1[or],out.yf[or,3],s1[or],out.yf[or,7],col=2)
#latitude
or<-order(s2)
ymin<-min(y,out.yf[,c(1,3,7)])
ymax<-max(y,out.yf[,c(1,3,7)])
par(mfrow=c(1,1))
plot(s2,y,ylim=c(ymin,ymax),xlab="latitude")
points(s2[or],out.yf[or,1],pch=20,col=2)
segments(s2[or],out.yf[or,3],s2[or],out.yf[or,7],col=2)
#
plot(y,out.yf[,1])
R2<-(cor(scallop$lgcatch,out.yf[,1]))^2
print(R2)
#map
map("usa",xlim=c(-74,-71),ylim=c(38.2,41.5))
int.scp<-interp(s1,s2,out.yf[,1])
contour(int.scp,add=TRUE)
image(int.scp,add=TRUE)

#Future predictions
out.yf2<-out.sum[grep("yf2",rownames(out.sum)),]
print(out.yf2[,c(1,3,7)])


#--- Ejemplo 12 ---
#install.packages("CARBayes")
#install.packages("INLA",repos=c(getOption("repos"),INLA="https://inla.r-inla-download.org/R/stable"), dep=TRUE)
#library(CARBayes)
#library(INLA)
install.packages("SpatialEpi")
install.packages("maps")
install.packages("maptools")
install.packages("RColorBrewer")
install.packages("classInt")
install.packages("spdep")
library(SpatialEpi)
library(maps)
library(maptools)
library(RColorBrewer)
library(classInt)
library(spdep)

#-Loading data-
data(scotland)
names(scotland)
head(scotland$data)
n<-nrow(scotland$data)

#Map of SMR
plotvar<-scotland$data$cases/scotland$data$expected
nclr<-7
plotclr<-brewer.pal(nclr,"YlOrBr")
class<-classIntervals(plotvar,nclr,dataPrecision=2,style="quantile")
colcode<-findColours(class,plotclr)
#
scotland.map <- scotland$spatial.polygon
plot(scotland.map,col=colcode)
legend(80,1200, legend=names(attr(colcode, "table")), 
       fill=attr(colcode, "palette"), cex=1, bty="n")
title(main="SMR")

#Map of AFF
plotvar<-scotland$data$AFF
nclr<-5
plotclr<-brewer.pal(nclr,"YlOrBr")
class<-classIntervals(plotvar,nclr,dataPrecision=2,style="quantile")
colcode<-findColours(class,plotclr)
#
scotland.map <- scotland$spatial.polygon
plot(scotland.map,col=colcode)
legend(80,1200, legend=names(attr(colcode, "table")), 
       fill=attr(colcode, "palette"), cex=1, bty="n")
title(main="AFF")


#-Defining data-
W.nb<-poly2nb(scotland$spatial.polygon)
print(W.nb)
m<-rep(0,n)
W.l<-matrix(NA,nrow=n,ncol=11)
adj<-NULL
for (i in 1:n) {
  if (W.nb[[i]][1]!=0) {
    m[i]<-length(W.nb[[i]])
    W.l[i,1:m[i]]<-W.nb[[i]]
    adj<-c(adj,W.nb[[i]])
  }}
W<-matrix(0,nrow=n,ncol=n)
for (i in 1:n) {
  for (j in 1:m[i]) {
    W[i,W.l[i,j]]<-1
    W[W.l[i,j],i]<-1
  }
}
weights<-rep(1,length(adj))
#
y<-scotland$data$cases
ee<-scotland$data$expected
x<-scotland$data$AFF
#
data<-list("n"=n,"y"=y,"ee"=ee,"x"=x,"adj"=adj,"weights"=weights,"num"=m)

#-Defining inits-
phi.i<-rep(0,n)
phi.i[6]<-NA
phi.i[8]<-NA
phi.i[11]<-NA
inits<-function(){list(beta=rep(0,2),tau.t=1,tau.c=1,theta=rep(0,n),phi=rep(0,n),yf=rep(0,n))}

#-Selecting parameters to monitor-
parameters<-c("beta","lambda","theta","phi","yf")

#-Running code-
#OpenBUGS
ej12.sim<-bugs(data,inits,parameters,model.file="Ej12.txt",
               n.iter=5000,n.chains=2,n.burnin=500,n.thin=1)
#JAGS
ej12.sim<-jags(data,inits,parameters,model.file="Ej11.txt",
               n.iter=5000,n.chains=2,n.burnin=500,n.thin=1)

#-Monitoring chain-

#Traza de la cadena
traceplot(ej12.sim)

#Cadena

#OpenBUGS
out<-ej12.sim$sims.list

#JAGS
out<-ej12.sim$BUGSoutput$sims.list

z<-out$beta[,2]
par(mfrow=c(2,2))
plot(z,type="l")
plot(cumsum(z)/(1:length(z)),type="l")
hist(z,freq=FALSE)
acf(z)

#Resumen (estimadores)
#OpenBUGS
out.sum<-ej12.sim$summary

#JAGS
out.sum<-ej12.sim$BUGSoutput$summary

#Tabla resumen
out.b<-out.sum[grep("beta",rownames(out.sum)),c(1,3,7)]
out.b<-cbind(out.b,apply(out$beta,2,prob))
dimnames(out.b)[[2]][4]<-"prob"
print(out.b)

#phi
out.phi<-out.sum[grep("phi",rownames(out.sum)),]
out.phi<-rbind(out.phi[1:5,],rep(0,ncol(out.phi)),out.phi[6:nrow(out.phi),])
out.phi<-rbind(out.phi[1:7,],rep(0,ncol(out.phi)),out.phi[8:nrow(out.phi),])
out.phi<-rbind(out.phi[1:10,],rep(0,ncol(out.phi)),out.phi[11:nrow(out.phi),])
out.est<-out.phi
k<-n
ymin<-min(out.est[,c(1,3,7)])
ymax<-max(out.est[,c(1,3,7)])
par(mfrow=c(1,1))
plot(1:k,out.est[,1],xlab="index",ylab="",ylim=c(ymin,ymax))
segments(1:k,out.est[,3],1:k,out.est[,7])
abline(h=0,col="grey70")
title("Efecto espacial")

#theta
out.the<-out.sum[grep("the",rownames(out.sum)),]
out.est<-out.the
k<-n
ymin<-min(out.est[,c(1,3,7)])
ymax<-max(out.est[,c(1,3,7)])
par(mfrow=c(1,1))
plot(1:k,out.est[,1],xlab="index",ylab="",ylim=c(ymin,ymax))
segments(1:k,out.est[,3],1:k,out.est[,7])
abline(h=0,col="grey70")
title("Efecto individual")

#DIC
out.dic<-ej12.sim$DIC
out.dic<-ej12.sim$BUGSoutput$DIC
print(out.dic)

#Predictions
out.yf<-out.sum[grep("yf",rownames(out.sum)),]
or<-order(y)
ymin<-min(y,out.yf[,c(1,3,7)])
ymax<-max(y,out.yf[,c(1,3,7)])
par(mfrow=c(1,1))
plot(y[or],ylim=c(ymin,ymax))
lines(out.yf[or,1],lwd=2,col=2)
lines(out.yf[or,3],lty=2,col=2)
lines(out.yf[or,7],lty=2,col=2)
#
plot(y,out.yf[,1])
R2<-(cor(y,out.yf[,1]))^2
print(R2)

#Map of lambda
out.lam<-out.sum[grep("lam",rownames(out.sum)),]
plotvar<-out.lam[,1]
nclr<-7
plotclr<-brewer.pal(nclr,"YlOrBr")
class<-classIntervals(plotvar,nclr,dataPrecision=2,style="quantile")
colcode<-findColours(class,plotclr)
#
scotland.map <- scotland$spatial.polygon
plot(scotland.map,col=colcode)
legend(80,1200, legend=names(attr(colcode, "table")), 
       fill=attr(colcode, "palette"), cex=1, bty="n")
title(main="CAR MR")
