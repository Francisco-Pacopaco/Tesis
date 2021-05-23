library(neuralnet)####Esta es la libreria necesaria para correr nuestra red neuronal
library(scatterplot3d)###Esta es la libreria para los graficos en 3D
####Estos son ejemplos para la clase de genomica
set.seed(42)
xdatos<-runif(200,-10,10)
y_target<-sin(xdatos)
base<-data.frame(xdatos,y_target)
######Esta es la función de activacion
sigmoidea<-function(t){
  1/(1+exp(-t))
}

#Esto es para indicarle que voy a dividir los datos en un 75% para entrenamiento
smp_size <- floor(0.75 * nrow(base))

train_ind <- sample(seq_len(nrow(base)), size = smp_size)

train_base <- base[train_ind, ]
test_base<- base[-train_ind, ]
nn0<-neuralnet(y_target ~ xdatos,train_base, linear.output = TRUE ,act.fct = tanh,hidden = 7 )
plot(nn0)

###Veamos que tan bueno es prediciendo valores
test_2<-subset(train_base, select = c("xdatos"))##Esto solo es para seleccionar la columnda de datos del test set
prediccion<-compute(nn0,test_2) ###Esto es para obtener los valores que la red predice
comparacion<-data.frame(real=train_base["y_target"], predicho=prediccion$net.result)
x11()
plot(train_base[,1],train_base[,2],col="blue")
points(train_base[,1],comparacion[,2], col="tomato")
####Ejemplo 2
Escherichia<-c(-1,-1,-1,-1,1,1,1,1)
Salmonella<-c(-1,-1,1,1,-1,-1,1,1)
Neisseria<-c(-1,1,-1,1,-1,1,-1,1)
good<-c(-1,-1,1,1,1,1,-1,-1)
datos<-data.frame(Escherichia,Salmonella,Neisseria,good)

##### Red neuronal
nn<-neuralnet(good ~ Escherichia+Salmonella+Neisseria,datos, linear.output = FALSE ,act.fct = sigmoidea)
###Estos son los resultados que predice la red
resultados<-data.frame(nn$net.result)
resultados<-2*round(resultados)-1
### Esta es la estructura de la red
plot(nn)
####### Intentemos con otra función de activación
softplus<-function(x){
  log(1+exp(x))
}
###Una nueva red
nn2<-neuralnet(good ~ Escherichia+Salmonella+Neisseria, datos ,linear.output = FALSE ,act.fct = softplus)
###Estos son los resultados de la nueva red
resultados2<-data.frame(nn2$net.result)
resultados2<-2*round(resultados2)-1

####
datos2<-((datos+1)/2)+1
colors<-c("#E69F00","#56B4E9")
colors<-colors[as.numeric(datos2$good)]
x11()
scatterplot3d(datos[,1:3],xlab="Escherichia",ylab="Salmonella",zlab="Neisseria",color = colors,pch = 19, grid=TRUE, main = "Datos")

#####Agregar mas neuronas a la primer red
nn3<-neuralnet(good ~ Escherichia+Salmonella+Neisseria,datos, linear.output = FALSE ,act.fct = sigmoidea, hidden = c(3,1))
resultados3<-data.frame(nn3$net.result)
resultados3<-2*round(resultados3)-1
plot(nn3)



