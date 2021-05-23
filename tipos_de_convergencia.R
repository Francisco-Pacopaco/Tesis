library(readr)
library(polyreg)
library(ie2misc)
#file.choose()
path_datos<- "C:\\Users\\franc\\Downloads\\auto-mpg.csv"
datos<-read.csv(path_datos,header = T, sep = ",") ###Esto es para cargar las bases de datos
autos<-na.omit(datos)
autos$origin <- as.numeric(as.character(autos$origin))
autos$horsepower <- as.numeric(as.character(autos$horsepower))
autos$X.mpg <- as.numeric(as.character(autos$X.mpg))
###Ahora hay que normalizar los datos, restar a cada columna su media y dividir entres sus desvianza
norm<-function(x){
  (x-mean(x))/sd(x)
}

mpg<-norm(autos$X.mpg)
cylinders<-norm(autos$cylinders)
displacement<-norm(autos$displacement)
horsepower<-norm(autos$horsepower)
weight<-norm(autos$weight)
acceleration<-norm(autos$acceleration)
origin<-norm(autos$origin)
model_year<-norm(autos$model.year)

autos2<-cbind(cylinders,displacement,horsepower,weight,acceleration,origin,model_year,mpg)
autos2<-data.frame(autos2)

#Ahora voy a partir los datos uno para entrenar y otro para validar, justo como en la red
set.seed(0) # Set Seed so that same sample can be reproduced in future also

# Ahora seleccionamos 80% de los datos como entrenamiento con semilla 0  
sample <- sample.int(n = nrow(autos2), size = floor(.8*nrow(autos2)), replace = F)
train <- autos2[sample, ]
test  <- autos2[-sample, ]
####Ahora hay que intentar hacer una regresión polinomial de mpg sobre las otr

modelo<-FSR(train, max_poly_degree=9, max_interaction_degree=9)
prediccion<-predict(modelo, test)

#MEan absolute value
mae(prediccion,test$mpg)

#####Con un polinomio de grado 3 y con terminos cruzados hasta 2 el mae es 0.272333
#####Con un polinomio de grado 3 y con terminos cruzados hasta 3 el mae es 0.2728657

#####Con un polinomio de grado 4 y con terminos cruzados hasta 4 el mae es 0.3834785
#####Con un polinomio de grado 4 y con terminos cruzados hasta 3 el mae es 0.24507
#####Con un polinomio de grado 4 y con terminos cruzados hasta 2 el mae es 0.310756
#####Con un polinomio de grado 4 y con terminos cruzados hasta 2 el mae es 0.0.3207452

#####Con un polinomio de grado 2 y con terminos cruzados hasta 1 el mae es 0.299024
#####Con un polinomio de grado 2 y con terminos cruzados hasta 2 el mae es 0.3906286

#####Con un polinomio de grado 1 y con terminos cruzados hasta 1 el mae es 0.3193175

#####Con un polinomio de grado 9 y con terminos cruzados hasta 9 el mae es 0.2943714

####Vamos a ver como se comporta con un modelo lineal
mod_lin<-lm(train$mpg ~ train$cylinders+train$displacement+train$horsepower+train$weight+train$acceleration+train$origin+train$model_year)
summary(mod_lin)

y_lin_hat<-0.0002372-0.0407212*test$cylinders+ 0.1671158*test$displacement-0.0822641*test$horsepower-0.6898938*test$weight+0.0450378*test$acceleration+0.1355932*test$origin+0.3611590*test$model_year

mae(y_lin_hat,test$mpg)

####Algunas funciones que convergen en l2 pero no hay convergencia uniforme
x<-seq(0,1, by= 0.0001)
f1<-function(x){
  x
}

f2 <- function(x,n){
  f<- NULL
  f[x< 1/n]<-n*x[x<1/n]
  f[1>x & x>= 1/n]<-1
  f[x>=1]<-1
  f
}

f3 <- function(x){
  f<- NULL
  f[x=0]<-0*x[x=0]
  f[1>x & x> 0]<-1
  f[x>=1]<-1
  f
}

plot(x, f2(x,2), type = 'l', col='tomato', lwd="2")
lines(x, f2(x,3), type = 'l', col='blue', lwd="2")
lines(x, f2(x,4), type = 'l', col='green', lwd="2")
lines(x,f2(x,5), type='l', col= 'purple', lwd="2")
lines(x,f3(x),type='l', col='orange', lwd="2")
legend("bottomright",c(expression(f_2),expression(f_3),expression(f_4),expression(f_5)),fill=c("tomato","blue","green","purple"))

###############Aproximaciones para las funciones de activación#####################
####Ejemplo de que la aproximación en L2 no implica la aproximación puntual

chr<-function(x){
  f<- NULL
  f[1>x & x> 0]<-1
  f[x>=1]<-0
  f[x<=0]<-0
  f
}

m<-3
k<-seq(0,(2**m)-1, by=1)

#Esta matriz es para guardar las diferentes funciones ya valuadas
secue<-function(x,a){
  chr((2**m)*x-a)
}

mat<-matrix(, nrow = length(x), ncol = length(k))
for(j in 1:(2**m)-1){
  mat[, j] <-secue(x,k[j])
}
###Esto solo es para escoger un color aleatorio, genera un vector con nombre de colores aleatorios
library(RColorBrewer)
n <- length(k)
qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
col_vector = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))
print(col_vector[1])
###Esto es para hacer las graficas

x11()
plot(x, chr(x), col='tomato', lwd="2", xlim=c(0,1), ylim=c(0.5,1.5), ylab = "y", main = "m=3", type = 'l')
points(x, secue(x,k[1])-(k[1]/100) , col=col_vector[1], lwd="2")
points(x, secue(x,k[2])-(k[2]/100) , col=col_vector[2], lwd="2")
points(x, secue(x,k[3])-(k[3]/100) , col=col_vector[3], lwd="2")
points(x, secue(x,k[4])-(k[4]/100) , col=col_vector[4], lwd="2")
points(x, secue(x,k[5])-(k[5]/100) , col=col_vector[5], lwd="2")
points(x, secue(x,k[6])-(k[6]/100) , col=col_vector[6], lwd="2")
points(x, secue(x,k[7])-(k[7]/100) , col=col_vector[7], lwd="2")
points(x, secue(x,k[8])-(k[8]/100) , col=col_vector[8], lwd="2")
points(x, secue(x,k2[1])-(k2[1]/100) , col=col_vector[9], lwd="2")
legend("bottomright",c(expression(f[1]),expression(f[2]),expression(f[3]),expression(f[4]),expression(f[5]),expression(f[6]), expression(f[7]), expression(f[8])),fill=c(col_vector[1],col_vector[2], col_vector[3], col_vector[4],col_vector[5],col_vector[6],col_vector[7],col_vector[8]))

m2<-2
k2<-seq(0,(2**m2)-1, by=1)
secue2<-function(x,a){
  chr((2**m2)*x-a)
}


x11()
plot(x, chr(x), col='tomato', lwd="0.5", xlim=c(0,1), ylim=c(0.5,1.5), ylab = "y", main = "m=2", type = 'l')
points(x, secue2(x,k2[1])-(k2[1]/100) , col=col_vector[1], lwd="2")
points(x, secue2(x,k2[2])-(k2[2]/100) , col=col_vector[2], lwd="2")
points(x, secue2(x,k2[3])-(k2[3]/100) , col=col_vector[3], lwd="2")
points(x, secue2(x,k2[4])-(k2[4]/100) , col=col_vector[4], lwd="2")
legend("bottomright",c(expression(f[1]),expression(f[2]),expression(f[3]),expression(f[4])),fill=c(col_vector[1],col_vector[2], col_vector[3], col_vector[4]))

