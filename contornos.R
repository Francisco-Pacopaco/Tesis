####MAs graficas para la tesis
library(lattice)
library(plotly)
library(ggplot2)
library(ContourFunctions)
library(shape)
#Graficas para las funciones de activación
t<-seq(from=-10,to=10,by=0.001)
sigmoid<-function(z){
  1/(1+exp(-z))
}
x11()
plot(t,sigmoid(t),col="green",type = "l",ylab = " ",xlab = "", lwd=2)
abline(h=0)
abline(v=0)

ReLU<-function(z){
  1/2*(z+abs(z))
}
x11()
plot(t,ReLU(t),col="purple",type = "l",ylab = " ",xlab = "", lwd=2)
abline(h=0)
abline(v=0)

x11()
plot(t,tanh(t),col="tomato",type = "l",ylab = " ",xlab = "",lwd=2)
abline(h=0)
abline(v=0)

mini<-function(z){
 (1/2)*(z+1-abs(1-z))
}

maxi<-function(z){
  (1/2)*(z-1+abs(z+1))
}

hardtanh<-function(z){
  maxi(mini(z))
}

x11()
plot(t,hardtanh(t),col="blue",type = "l",ylab = " ",xlab = "", lwd=2)
abline(h=0)
abline(v=0)

#Este es el paraboliode
paraboliode<-function(x,y){
  (2/3)*x^2+2*y^2
}
x<-seq(-1,1, length=20)
y<-seq(-1,1, length=20)
z<-outer(x,y,paraboliode)
x11()
wireframe(z, data = NULL,
           drape = TRUE,,
           xlab=expression('w'[1]),
           ylab=expression('w'[2]),
           colorkey=FALSE,
           zlab='Error',
           main='Función de perdida',
           scales = list( arrows = FALSE, col="black"),
           par.settings = list(axis.line = list(col = 'transparent')),
          strip.border = list(col = 'transparent'),
          strip.background = list(col = 'transparent'))
                              
           
x11()
cf_grid(x, y, z, lines_only=TRUE, 
        bar = FALSE, 
        main='Contornos de nivel',
        levels = c(0.2,0.4,0.6,1,1.4,1.8,2.2))
points(0.5,0.5,pch=19,col='tomato')
text(0.5,0.46,"mínimo")
Arrows(0.2,0.2,0.2,0.24,arr.type="triangle",arr.width=0.2,col = "aquamarine3")
Arrows(0.2,0.27,0.24,0.32,arr.type="triangle",arr.width=0.2,col = "aquamarine3")
Arrows(0.26,0.35,0.31,0.4,arr.type="triangle",arr.width=0.2,col = "aquamarine3")

x11()
cf_grid(x, y, z, lines_only=TRUE, 
        bar = FALSE, 
        main='Contornos de nivel',
        levels = c(0.2,0.4,0.6,1,1.4,1.8,2.2))
points(0.5,0.5,pch=19,col='tomato')
text(0.5,0.46,"mínimo")
Arrows(0.1,0.1,0.3,0.6,arr.type="triangle",arr.width=0.2,col = "chocolate3")
Arrows(0.3,0.6,0.7,0.15,arr.type="triangle",arr.width=0.2,col = "chocolate3")
Arrows(0.7,0.15,0.9,0.7,arr.type="triangle",arr.width=0.2,col = "chocolate3")

#######Funcion
timo<-function(a,b,c){
  (a+4*b+c)*(1/6)
}
timo2<-function(a,b){
  ((b-a)*(1/6))**2
}

print(timo(1,3,5))
print(timo2(1,5))
#####
contorno<-function(x,y){
  800+10*x+7*y-8.5*(x**2)-5*(y**2)+4*x*y
}
x1<-seq(0,10, length=100)
y1<-seq(0,10, length=100)
z1<-outer(x1,y1,contorno)

x11()
cf_grid(x1, y1, z1, lines_only=TRUE, 
        bar = FALSE, 
        main='Contornos de nivel',
        levels = c(800,750,700,625,550,475,400,325,250,175,100,25))
####Modelo polinomial varias variables
modelop<-function(T,C){
  -1105.56+8.0242*T+22.994*C-0.0142*T**2-0.20502*C**2-0.062*T*C
}

x2<-seq(180,260, length=100)
y2<-seq(15,30, length=100)
z2<-outer(x2,y2,modelop)


x11()
cf_grid(x2, y2, z2, lines_only=TRUE, 
        bar = FALSE, 
        main='Contornos de nivel',
        levels = c(80,75,70,62.5,55,47.5,40,32.5,25,17.5,10))
x11()
wireframe(z2, data = NULL,
          drape = TRUE,,
          xlab=expression('T'),
          ylab=expression('C'),
          colorkey=FALSE,
          zlab='y',
          main='Superficie de Respuesta',
          scales = list( arrows = FALSE, col="black"),
          par.settings = list(axis.line = list(col = 'transparent')),
          strip.border = list(col = 'transparent'),
          strip.background = list(col = 'transparent'))
temperatura<-c(200,250,200,250, 189.65, 260.35,225,225,225,225,225,225)
concentracion<-c(15,15,25,25,20,20,12.93,27.07,20,20,20,20)
respuesta<-c(43,78,69,73,48,76,65,74,76,79,83,81)
unos<-c()
for (i in 1:length(temperatura)) {
  unos[i]=1
  
}
matriz<-cbind(unos,temperatura,concentracion,temperatura**2,concentracion**2,temperatura*concentracion)
beta<-t(matriz)%*%matriz
beta<-solve(beta)
beta<-beta%*%t(matriz)%*%respuesta
B<-c(-0.0142,-0.062/2)
B1<-c(-0.062/2,-0.20502)
B<-cbind(B,B1)
lin<-c(8.0242,22.994)
pt_cr<-(-1/2)*solve(B)%*%lin
val_cr<-beta[1]+(1/2)*t(pt_cr)%*%lin
pro<-eigen(B)
val_pro<-pro$values
vec_pro<-pro$vectors
norm_vec <- function(x) sqrt(sum(x^2))
print(norm_vec(vec_pro[,1]))
temp1<-temperatura-pt_cr[1,1]
conc1<-concentracion-pt_cr[2,1]
aux1<-cbind(temp1,conc1)
w<-t(vec_pro)%*%t(aux1)
print(val_pro)
