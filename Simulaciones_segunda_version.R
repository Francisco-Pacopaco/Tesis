###Primero hayq eu cargar las librerias necesarias
library(gtools)
library(neuralnet)
library(pracma)
library(mvtnorm)
library(ggplot2)
library(cowplot)

###Funciones de activación
sigmoide<-function(x){
  1/(1+exp(-x))
}
softplus<-function(x){
  log(1+ exp(x))
}

mi_tanh<-function(x){
  tanh(x)
}
###Esta es una versión en r del métodoo de aproximación de Chebyshev
###Hiperparametros
n<-3 ###El grado de aproximación del polinomio
m<-4 ##El número de nodos o raices
x_min<--1 #Estos son los extremos del intervalo sobre el cual se hace la aproximación
x_max<- 1 #Estos son los extremos del intervalo sobre el cual se hace la aproximación


x_grid<-seq(x_min,x_max, by= .001)

scale_up<-function(z,x_min,x_max){ ##Escala un número entre -1 y 1 al intervalo [x_min, x_max]
  x_min+((z+1)*(x_max-x_min)/2)
}

scale_down<-function(x,x_min,x_max){ ##Escala un número entre [x_min, x_max] y lo mapea entre -1 y 1
  2*(x-x_min)/(x_max-x_min)-1
}

aux1<-seq(1,m,by=1)
r_k<--cos((2*aux1-1)*pi/(2*m)) ###Estas son las raices de los polinomios ortogonales de Chebyshev

###Generar la matriz de Vandermonde
unos<-c()
for (i in 1:(n+1)) {
  unos[i]=1
  
}
T<-matrix(,m,n+1)
T[,1]<-t(unos)
T[,2]<-t(r_k)
for (j in 2:n) {
  T[,j+1]= 2*t(r_k)*T[,j]-T[,j-1]
  
}

####Calcular los coeficientes
x_k<-scale_up(r_k,x_min,x_max)
y_k<-sigmoide(x_k) ##En este caso se esta usando la función sigmoide como ejemplo 
alfa<-solve(t(T)%*%T)%*%t(T)%*%y_k

###Calcular la aproximación de f, 
unos<-c()
for (j in 1:length(x_grid)) {
  unos[j]=1
  
}
T<-matrix(,length(x_grid),n+1)
T[,1]<-t(unos)
z_grid<-scale_down(x_grid,x_min,x_max)
T[,2]<-t(z_grid)
for (j in 2:n) {
  T[,j+1]= 2*t(z_grid)*T[,j]-T[,j-1]
  
}

##Aproximacion
Tf<-T%*%alfa
###Esto solo es una prueba
x11()
plot(x_grid,sigmoide(x_grid),col='tomato')
points(x_grid,Tf, col='blue')

###Esta función sirve para generar los datos con los datos con los que se va a trabajar
#####Esto es para generar los datos 
generateNormalData <- function(n_sample, p, q_original, mean_range, beta_range, error_var) {
  
  # Obtain the values for the means unformly distributed in the given range
  mean_values <- runif(p, mean_range[1], mean_range[2])
  
  X <- rmvnorm(n_sample, mean_values) #Para genetar datos de una distribución normal multivariada, el segundo argumento es el vector de medias 
  
  
  # compute the needed number of betas with
  n_betas <- 0
  for (t in 0:q_original) {
    n_betas <- n_betas + choose(p + t - 1, t) # each time adding the number of possible combinations with repetition with length t
  }
  
  # Obtain the values for thebetas unformly distributed in the given range
  original_betas <- runif(n_betas, beta_range[1], beta_range[2])
  
  
  # intialize the response vector
  Y <- rep(0, n_sample)
  # set a counter
  
  
  # loop over all the sample
  for (i in 1:n_sample) {
    # set up counter to know which beta we are using at each step
    counter <- 1
    
    Y[i] <- original_betas[1] # add the intercept first
    
    for (t in 1:q_original) {
      # Compute the possible combinations of length t and store the number of them.
      indexes <- combinations(p, t, repeats.allowed = TRUE) # needs library(gtools)
      indexes.rows <- nrow(indexes)
      
      # loop over all combinations of length t
      for (ind in 1:indexes.rows) {
        # product of all the variables for a given combination
        product <- 1
        for (j in 1:length(indexes[ind, ])) {
          product <- product * X[i, indexes[ind, j]]
        }
        
        # add each term to the response
        Y[i] <- Y[i] + original_betas[counter + ind] * product
      }
      # update counter after all combinations of length t are computed
      counter <- counter + indexes.rows
    }
  }
  
  # finally we add some normal errors:
  Y <- Y + rnorm(n_sample, 0, error_var)
  
  # Store all as a data frame
  data <- as.data.frame(cbind(X, Y))
  
  # Output includes the data and the original betas to comapre later
  output <- vector(mode = "list", length = 2)
  output[[1]] <- data
  output[[2]] <- original_betas
  names(output) <- c("data", "original_betas")
  return(output)
}

####Esta funcion divide los datos en datos de entrenamiento y datos de test

divideTrainTest <- function(data, train_proportion) {
  index <- sample(1:nrow(data), round(train_proportion * nrow(data)))
  train <- data[index, ]
  test <- data[-index, ]
  
  output <- vector(mode = "list", length = 2)
  output[[1]] <- train
  output[[2]] <- test
  names(output) <- c("train", "test")
  
  return(output)
}


####ahora hay que reescalar los datos con esta función
ScaleData2 <- function(data, scale_method = "0,1") {
  
  if (scale_method == "0,1") {
    #### Scale the data in the [0,1] interval and separate train and test ####
    
    maxs <- apply(data, 2, max) # obtain the max of each variable
    mins <- apply(data, 2, min) # obtain the min of each variable
    output <- as.data.frame(scale(data, center = mins, scale = maxs - mins)) #Resta el valor de la variable 'center' a cada columna y divide entre el valor 'scale' en cada columna también
    
  } else if (scale_method == "-1,1") {
    #### Scale the data in the [-1,1] interval and separate train and test ####
    
    maxs <- apply(data, 2, max) # obtain the max of each variable
    mins <- apply(data, 2, min) # obtain the min of each variable
    output <- as.data.frame(scale(data, center = mins + (maxs - mins) / 2, scale = (maxs - mins) / 2))
    
  } else if (scale_method == "standardize") {
    #### Scale the data to have mean=0 and sd=1 and separate train and test ####
    
    output <- as.data.frame(scale(data, center = TRUE, scale = TRUE))
    
  } else {
    
    print("Non valid method")
    output <- "Non valid method"
    
  }
  
  return(output)
  
}

###Estos son los coeficientes que se obtienen aplicando el método de Chebyshev
coeficientes<-function(w,v,alfa){
  aux1_beta0<-v[1]+alfa[1]*(v[2]+v[3]+v[4]+v[5])-alfa[3]*(v[2]+v[3]+v[4]+v[5])
  aux2_beta0<-v[2]*w[1,1]*(alfa[2]-3*alfa[4]+2*alfa[3]*w[1,1]+4*alfa[4]*w[1,1]*w[1,1])
  aux3_beta0<-v[3]*w[2,1]*(alfa[2]-3*alfa[4]+2*alfa[3]*w[2,1]+4*alfa[4]*w[2,1]*w[2,1])
  aux4_beta0<-v[4]*w[3,1]*(alfa[2]-3*alfa[4]+2*alfa[3]*w[3,1]+4*alfa[4]*w[3,1]*w[3,1])
  aux5_beta0<-v[5]*w[4,1]*(alfa[2]-3*alfa[4]+2*alfa[3]*w[4,1]+4*alfa[4]*w[4,1]*w[4,1])
  beta_0<-aux1_beta0+aux2_beta0+aux3_beta0+aux4_beta0+aux5_beta0
  aux1_beta1<-w[1,2]*v[2]*(alfa[2]-3*alfa[4]+4*alfa[3]*w[1,1]+12*alfa[4]*w[1,1]*w[1,1])
  aux2_beta1<-w[2,2]*v[3]*(alfa[2]-3*alfa[4]+4*alfa[3]*w[2,1]+12*alfa[4]*w[2,1]*w[2,1])
  aux3_beta1<-w[3,2]*v[4]*(alfa[2]-3*alfa[4]+4*alfa[3]*w[3,1]+12*alfa[4]*w[3,1]*w[3,1])
  aux4_beta1<-w[4,2]*v[5]*(alfa[2]-3*alfa[4]+4*alfa[3]*w[4,1]+12*alfa[4]*w[4,1]*w[4,1])
  beta_1<-aux1_beta1+aux2_beta1+aux3_beta1+aux4_beta1
  aux1_beta11<-v[2]*w[1,2]*w[1,2]*(2*alfa[3]+12*alfa[4]*w[1,1])
  aux2_beta11<-v[3]*w[2,2]*w[2,2]*(2*alfa[3]+12*alfa[4]*w[2,1])
  aux3_beta11<-v[4]*w[3,2]*w[3,2]*(2*alfa[3]+12*alfa[4]*w[3,1])
  aux4_beta11<-v[5]*w[4,2]*w[4,2]*(2*alfa[3]+12*alfa[4]*w[4,1])
  beta_11<-aux1_beta11+aux2_beta11+aux3_beta11+aux4_beta11
  beta_111<-4*alfa[4]*(v[2]*(w[1,2])^3+v[3]*(w[2,2])^3+v[4]*(w[3,2])^3+v[5]*(w[4,2])^3)
  aux1_beta2<-w[1,3]*v[2]*(alfa[2]-3*alfa[4]+4*alfa[3]*w[1,1]+12*alfa[4]*w[1,1]*w[1,1])
  aux2_beta2<-w[2,3]*v[3]*(alfa[2]-3*alfa[4]+4*alfa[3]*w[2,1]+12*alfa[4]*w[2,1]*w[2,1])
  aux3_beta2<-w[3,3]*v[4]*(alfa[2]-3*alfa[4]+4*alfa[3]*w[3,1]+12*alfa[4]*w[3,1]*w[3,1])
  aux4_beta2<-w[4,3]*v[5]*(alfa[2]-3*alfa[4]+4*alfa[3]*w[4,1]+12*alfa[4]*w[4,1]*w[4,1])
  beta_2<-aux1_beta2+aux2_beta2+aux3_beta2+aux4_beta2
  aux1_beta12<-w[1,2]*w[1,3]*v[2]*(4*alfa[3]+24*alfa[4]*w[1,1])
  aux2_beta12<-w[2,2]*w[2,3]*v[3]*(4*alfa[3]+24*alfa[4]*w[2,1])
  aux3_beta12<-w[3,2]*w[3,3]*v[4]*(4*alfa[3]+24*alfa[4]*w[3,1])
  aux4_beta12<-w[4,2]*w[4,3]*v[5]*(4*alfa[3]+24*alfa[4]*w[4,1])
  beta_12<-aux1_beta12+aux2_beta12+aux3_beta12+aux4_beta12
  beta_112<-12*alfa[4]*(w[1,3]*v[2]*(w[1,2])^2+w[2,3]*v[3]*(w[2,2])^2+w[3,3]*v[4]*(w[3,2])^2+w[4,3]*v[5]*(w[4,2])^2)
  aux1_beta22<-v[2]*w[1,3]*w[1,3]*(2*alfa[3]+12*alfa[4]*w[1,1])
  aux2_beta22<-v[3]*w[2,3]*w[2,3]*(2*alfa[3]+12*alfa[4]*w[2,1])
  aux3_beta22<-v[4]*w[3,3]*w[3,3]*(2*alfa[3]+12*alfa[4]*w[3,1])
  aux4_beta22<-v[5]*w[4,3]*w[4,3]*(2*alfa[3]+12*alfa[4]*w[4,1])
  beta_22<-aux1_beta22+aux2_beta22+aux3_beta22+aux4_beta22
  beta_122<-12*alfa[4]*(w[1,2]*v[2]*(w[1,3])^2+w[2,2]*v[3]*(w[2,3])^2+w[3,2]*v[4]*(w[3,3])^2+w[4,2]*v[5]*(w[4,3])^2)
  beta_222<-4*alfa[4]*(v[2]*(w[1,3])^3+v[3]*(w[2,3])^3+v[4]*(w[3,3])^3+v[5]*(w[4,3])^3)
  aux1_beta3<-w[1,4]*v[2]*(alfa[2]-3*alfa[4]+4*alfa[3]*w[1,1]+12*alfa[4]*w[1,1]*w[1,1])
  aux2_beta3<-w[2,4]*v[3]*(alfa[2]-3*alfa[4]+4*alfa[3]*w[2,1]+12*alfa[4]*w[2,1]*w[2,1])
  aux3_beta3<-w[3,4]*v[4]*(alfa[2]-3*alfa[4]+4*alfa[3]*w[3,1]+12*alfa[4]*w[3,1]*w[3,1])
  aux4_beta3<-w[4,4]*v[5]*(alfa[2]-3*alfa[4]+4*alfa[3]*w[4,1]+12*alfa[4]*w[4,1]*w[4,1])
  beta_3<-aux1_beta3+aux2_beta3+aux3_beta3+aux4_beta3
  aux1_beta13<-w[1,2]*w[1,4]*v[2]*(4*alfa[3]+24*alfa[4]*w[1,1])
  aux2_beta13<-w[2,2]*w[2,4]*v[3]*(4*alfa[3]+24*alfa[4]*w[2,1])
  aux3_beta13<-w[3,2]*w[3,4]*v[4]*(4*alfa[3]+24*alfa[4]*w[3,1])
  aux4_beta13<-w[4,2]*w[4,4]*v[5]*(4*alfa[3]+24*alfa[4]*w[4,1])
  beta_13<-aux1_beta13+aux2_beta13+aux3_beta13+aux4_beta13
  beta_113<-12*alfa[4]*(w[1,4]*v[2]*(w[1,2])^2+w[2,4]*v[3]*(w[2,2])^2+w[3,4]*v[4]*(w[3,2])^2+w[4,4]*v[5]*(w[4,2])^2)
  aux1_beta23<-w[1,3]*w[1,4]*v[2]*(4*alfa[3]+24*alfa[4]*w[1,1])
  aux2_beta23<-w[2,3]*w[2,4]*v[3]*(4*alfa[3]+24*alfa[4]*w[2,1])
  aux3_beta23<-w[3,3]*w[3,4]*v[4]*(4*alfa[3]+24*alfa[4]*w[3,1])
  aux4_beta23<-w[4,3]*w[4,4]*v[5]*(4*alfa[3]+24*alfa[4]*w[4,1])
  beta_23<-aux1_beta23+aux2_beta23+aux3_beta23+aux4_beta23
  beta_123<-24*alfa[4]*(v[2]*w[1,2]*w[1,3]*w[1,4]+v[3]*w[2,2]*w[2,3]*w[2,4]+v[4]*w[3,2]*w[3,3]*w[3,4]+v[5]*w[4,2]*w[4,3]*w[4,4])
  beta_223<-12*alfa[4]*(v[2]*w[1,4]*(w[1,3])^2+v[3]*w[2,4]*(w[2,3])^2+v[4]*w[3,4]*(w[3,3])^2+v[5]*w[4,4]*(w[4,3])^2)
  aux1_beta33<-v[2]*w[1,4]*w[1,4]*(2*alfa[3]+12*alfa[4]*w[1,1])
  aux2_beta33<-v[3]*w[2,4]*w[2,4]*(2*alfa[3]+12*alfa[4]*w[2,1])
  aux3_beta33<-v[4]*w[3,4]*w[3,4]*(2*alfa[3]+12*alfa[4]*w[3,1])
  aux4_beta33<-v[5]*w[4,4]*w[4,4]*(2*alfa[3]+12*alfa[4]*w[4,1])
  beta_33<-aux1_beta33+aux2_beta33+aux3_beta33+aux4_beta33
  beta_133<-12*alfa[4]*(v[2]*w[1,2]*(w[1,4])^2+v[3]*w[2,2]*(w[2,4])^2+v[4]*w[3,2]*(w[3,4])^2+v[5]*w[4,2]*(w[4,4])^2)
  beta_233<-12*alfa[4]*(v[2]*w[1,3]*(w[1,4])^2+v[3]*w[2,3]*(w[2,4])^2+v[4]*w[3,3]*(w[3,4])^2+v[5]*w[4,3]*(w[4,4])^2)
  beta_333<-4*alfa[4]*(v[2]*(w[1,4])^3+v[3]*(w[2,4])^3+v[4]*(w[3,4])^3+v[5]*(w[4,4])^3)
  return(c(beta_0,beta_1,beta_2,beta_3,beta_11,beta_22,beta_33,beta_12,beta_13,beta_23,beta_112,beta_113,beta_122,beta_123,beta_133,beta_223,beta_233,beta_111,beta_222,beta_333))
}



####Ahora vamos a comparar con el méetodo del autor 


#####Funcion para obtener los coeficientes cuya formula se encuentra en el articulo del autor 
##El parámetro g son los coeficientes de la aproximaciónde Taylor de la función de activación
obtainCoeffsFromWeights=function(w,v,g){
  # w matrix of size h_1*p+1, such that the elements are w_ji
  # v vector of length h_1+1 such that the elements are v_j
  # g vector of length q+1 such that g=(g(0),g'(0),g''(0),...,g^{(q)}(0))
  
  # To follow our original notation we need to set uo the values for q,p and h_1. 
  # note that this is because the vector starts at 0 in our notation
  
  q=length(g)-1 
  h_1=dim(w)[1] 
  p=dim(w)[2]-1
  
  # We define the vector that will contain all the coefficents (betas) and their associated indexes (labels)
  
  betas=c(0)
  labels=c("0")
  
  #Now we apply the formulas described previously.
  
  # beta 0 (special case)
  
  betas[1]=v[1] # first get v[1]=v_0 in our notation
  
  # Now obtain the summation:
  for (j in 1:h_1){
    aux=0 # this auxiliar variable will store the inner summation
    for (n in 0:q){
      aux=aux+g[n+1]*w[j,1]^n # we have to use g[n+1] to obtain g^(n)/n!, because the function taylor already includes the term 1/n!
    }
    betas[1]=betas[1]+v[j+1]*aux # we have to use v[j+1] to obtain v_j
  }
  
  
  # The rest of the betas:
  
  #For each t from 1 to q, wehre t is the number of subindexes in the coefficent:
  
  for (t in 1:q){
    
    #We need to find all the possible combinations (order does not matter) of length t with elements from 1 to p (the input variables) with repetition.
    indexes <- combinations(p,t,repeats.allowed = TRUE) # needs library(gtools)
    indexes.rows=nrow(indexes) #store the number of all possible combinations for a given t
    
    #Now we create temporal betas and labels for the given t that we will include in the final result vector
    betas_t=rep(0,indexes.rows)
    labels_betas_t=rep("label",indexes.rows)
    
    # Loop over all the number of possible combinations
    for(combination_index in 1:indexes.rows){
      
      #Obtain a vector containing the indexes, l_1,l_2,...,l_t in our notation:
      l_values=indexes[combination_index,]
      
      #Create the label as a string of the form "l_1,l_2,...,l_t"
      labels_betas_t[combination_index]=paste(as.character(indexes[combination_index,]),collapse = ",")
      
      #Now we can obtain the vector m=(m_1,...,m_p). The value for m_0 changes with n and will be added later in the summation
      m=rep(0,p)
      for (i in 1:p){
        m[i]=sum(l_values==i)
      }
      
      #Finally we can apply the general formula:
      for (j in 1:h_1){
        aux=0
        for (n in t:q){
          m_n=c(n-t,m)
          aux=aux+g[n+1]*factorial(n)/prod(factorial(m_n))*prod(w[j,]^m_n)
        }
        betas_t[combination_index]=betas_t[combination_index]+v[j+1]*aux
      }
    }
    
    #Include all the values and the labels for each t in the final vector 
    betas=c(betas,betas_t)
    labels=c(labels,labels_betas_t)
  }
  
  #Finally set the betas vector as a row matrix and use the labels as the column names
  betas=t(as.matrix(betas))
  colnames(betas)=labels
  return(betas)
}

#Estos son los coeficientes que se obtienen con método del autor 

coeff <- obtainCoeffsFromWeights(w, v, g)

###Este evalua la regresión polinomial
evaluatePR <- function(x,betas) {
  #performs the result of the polynomial regression expresion given the betas and their labels.
  x=unname(as.matrix(x)) #removes the colnames and rownames of the input variables when using a dataframe.
  
  response=betas[1] # this gets the intercept beta_0
  for (i in 2:length(betas)){
    #here the label is transformed into a vector of the needed length with the index of each variable
    variable_indexes=as.integer(unlist(strsplit(colnames(betas)[i], ",")))
    
    #Intialize the product as 1 and loop over all the indexes l_j to obtain the product of al the needed variables x
    product=1
    for(j in 1:length(variable_indexes)){
      product=product*x[variable_indexes[j]]
    }
    #We add to the response the product of those variables with their associated beta
    response=response+betas[i]*product
  }
  return(response)
}
# Parameters for the data generation
n_simulation <- 500 ###Este es el número de simulaciones
n_sample <- 200 ##Este es el tamaño de muestra en cada simulación
p <- 3 #Este es la dimensión del vector de atributos
q_original <- 2 ##Este es el grado del polinomio de la variable objetivo
mean_range <- c(-10, 10) ###Este es el rango de la media que sirve para generar los datos, se usa en la normal multivariada
beta_range <- c(-5, 5) ##Este es el rango donde los coeficientes beta 
error_var <- 0.1
h_1<-4
stepmax<-1e+05

###Esta función ejecuta la regresión utilizando los coeficientes de chebyshev
####Ahora hay que programar la regresion
aux_unos<-c()
for (i in 1:n_sample*(1/4)) {
  aux_unos[i]=1
}
poli_regre<-function(data,beta){
  p1<-beta[1]*t(aux_unos)+beta[2]*data[,1]+beta[3]*data[,2]+beta[4]*data[,3]
  p2<-beta[5]*(data[,1])^2+beta[6]*(data[,2])^2+beta[7]*(data[,3])^2
  p3<-beta[8]*data[,1]*data[,2]+beta[9]*data[,1]*data[,3]+beta[10]*data[,2]*data[,3]
  p4<-beta[11]*data[,1]*data[,1]*data[,2]+beta[12]*data[,1]*data[,1]*data[,3]+beta[13]*data[,1]*data[,2]*data[,2]
  p5<-beta[14]*data[,1]*data[,2]*data[,3]+beta[15]*data[,1]*data[,3]*data[,3]+beta[16]*data[,2]*data[,2]*data[,3]
  p6<-beta[17]*data[,2]*data[,3]*data[,3]
  p7<-beta[18]*data[,1]*data[,1]*data[,1]+beta[19]*data[,2]*data[,2]*data[,2]+beta[20]*data[,3]*data[,3]*data[,3]
  return(p1+p2+p3+p4+p5+p6+p7)
}


###Ahora hay que hacer la función que ejecute todo un ejemplo

ejecutar_ejemplo<-function(n_sample, p, q_original, mean_range, beta_range, error_var, scale_method, h_1, fun, q_taylor, stepmax = 1e+05){
  # Generate the data:
  data_generated <- generateNormalData(n_sample, p, q_original, mean_range, beta_range, error_var)
  data <- data_generated$data
  original_betas <- data_generated$original_betas
  
  #### Scale the data in the desired interval and separate train and test ####
  
  data_scaled <- ScaleData2(data, scale_method)
  
  aux <- divideTrainTest(data_scaled, train_proportion = 0.75)
  
  train <- aux$train
  test <- aux$test
 
  
  # To use neural net we need to create the formula as follows, Y~. does not work. This includes all the variables X:
  var.names <- names(train)
  formula <- as.formula(paste("Y ~", paste(var.names[!var.names %in% "Y"], collapse = " + ")))
  
  # train the net:
  nn <- neuralnet(formula, data = train, hidden = h_1, linear.output = T, act.fct = fun, stepmax = stepmax)
  
  #Esta es la regresión polinomial
  regre_poli<-lm(Y~V1+ V2+V3+ V1*V2+ V1*V3+ V3*V2 + V1*V2*V3+V1*V1+V2*V2+ V3*V3, data = train)
  predic_regre_poli<-predict(regre_poli,test)
  
  # obtain the weights from the NN model:
  w <- t(nn$weights[[1]][[1]]) # we transpose the matrix to match the desired dimensions
  v <- nn$weights[[1]][[2]]
  
  # Obtain the vector with the derivatives of the activation function up to the given degree:
  g <- rev(taylor(fun, 0, q_taylor))
  
  ###Hiperparametros
  n<-q_taylor ###El grado de aproximación del polinomio
  m<-q_taylor+1 ##El número de nodos o raices
  
  aux1<-seq(1,m,by=1)
  r_k<--cos((2*aux1-1)*pi/(2*m))
  
  ###Generar la matriz de Vandermonde
  unos<-c()
  for (i in 1:(n+1)) {
    unos[i]=1
    
  }
  T<-matrix(,m,n+1)
  T[,1]<-t(unos)
  T[,2]<-t(r_k)
  for (j in 2:n) {
    T[,j+1]= 2*t(r_k)*T[,j]-T[,j-1]
    
  }
  x_min<--1
  x_max<-1
  ####Calcular los coeficientes
  x_k<-scale_up(r_k,x_min,x_max)
  y_k<-fun(x_k)
  alfa<-solve(t(T)%*%T)%*%t(T)%*%y_k
  
  betas1<-coeficientes(w,v,alfa)
  prediccion_poli<-poli_regre(test,betas1)
  prediccion_poli<-t(prediccion_poli)
  
  # Apply the formula
  coeff <- obtainCoeffsFromWeights(w, v, g)
  
  # Obtain the predicted values for the test data with our Polynomial Regression
  n_test <- length(test$Y)
  PR.prediction <- rep(0, n_test)
  
  for (i in 1:n_test) {
    PR.prediction[i] <- evaluatePR(test[i, seq(p)], coeff) ####Este es el metodo del autor 
  }
  
  # Obtain the predicted values with the NN
  NN.prediction <- predict(nn, test)
  
  # MSE:
  
  NN.MSE <- sum((test$Y - NN.prediction)^2) / n_test
  PR.MSE <- sum((test$Y - PR.prediction)^2) / n_test
  PC.MSE <- sum((test$Y - prediccion_poli)^2) / n_test
  poli_regre<- sum((test$Y - predic_regre_poli)^2) / n_test
  
  # MSE between NN and PR (because PR is actually approximating the NN, not the actual response Y)
  MSE.NN.vs.PR <- sum((NN.prediction - PR.prediction)^2) / n_test
  MSE.NN.vs.PC <- sum((NN.prediction - prediccion_poli)^2) / n_test
  MSE.NN.vs.poli_regre<- sum((NN.prediction - predic_regre_poli)^2) / n_test
  
  
  output <- vector(mode = "list", length = 18)
  output[[1]] <- train
  output[[2]] <- test
  output[[3]] <- g
  output[[4]] <- nn
  output[[5]] <- coeff
  output[[6]] <- betas1
  output[[7]] <- alfa
  output[[8]] <- NN.prediction
  output[[9]] <- PR.prediction
  output[[10]] <- prediccion_poli
  output[[11]] <- NN.MSE
  output[[12]] <- PR.MSE
  output[[13]]<-PC.MSE
  output[[14]]<-MSE.NN.vs.PR
  output[[15]]<-MSE.NN.vs.PC
  output[[16]]<-original_betas
  output[[17]]<-predic_regre_poli
  output[[18]]<-MSE.NN.vs.poli_regre
  
  names(output) <- c("train", 
                     "test", 
                     "coeff_taylor_activacion", 
                     "nn", 
                     "coeff_articulo", 
                     "coeff_Cheby",
                     "coeff_Cheby_funcion_activacion",
                     "NN.prediction", 
                     "PR.prediction", 
                     "Cheby.prediction",
                     "NN.MSE", 
                     "PR.MSE", 
                     "Pc.MSE", 
                     "MSE.NN.VS.PR", 
                     "MSE.NN.vs.PC",
                     "original_betas",
                     "regresion_polinomial_prediccion",
                     "MSE.NN.vs.poli_regre")
  
  
  return(output)
  
}
###Esta es la semilla 
set.seed(12345)



####Simulaciones funcion softplus sobre el intervalo [0,1]
simulations_MSE_softplus_01<-matrix(, nrow = n_simulation, ncol = 3)
for (i in 1:n_simulation ) {
  ejemplo<-ejecutar_ejemplo(n_sample, p, q_original, mean_range, beta_range, error_var, scale_method = "0,1",h_1, softplus, q_taylor=3, stepmax)
  simulations_MSE_softplus_01[i,1]<-ejemplo$MSE.NN.vs.PC
  simulations_MSE_softplus_01[i,2]<-ejemplo$MSE.NN.VS.PR
  simulations_MSE_softplus_01[i,3]<-ejemplo$MSE.NN.vs.poli_regre
}

#Softplus_01_Chebyshev
print(mean(simulations_MSE_softplus_01[,1]))
print(var(simulations_MSE_softplus_01[,1]))
x11()
boxplot(simulations_MSE_softplus_01[,1], horizontal = TRUE, ylim=c(0,0.1), col = "orange", main="Función de Activación Softplus")
legend("topright",c('media=1.009111',"varianza=145.082","Chebyshev=3"))

#Softplus_01_Taylor
print(mean(simulations_MSE_softplus_01[,2]))
print(var(simulations_MSE_softplus_01[,2]))
x11()
boxplot(simulations_MSE_softplus_01[,2], horizontal = TRUE, ylim=c(0,0.1), col = "tomato", main="Función de Activación Softplus")
legend("topright",c('media=1.196432',"varianza=145.197.1704","Taylor=3"))

#Softplus_01_Regre_poli
print(mean(simulations_MSE_softplus_01[,3]))
print(var(simulations_MSE_softplus_01[,3]))
x11()
boxplot(simulations_MSE_softplus_01[,3], horizontal = TRUE, ylim=c(0,0.01), col = "yellow", main="Función de Activación Softplus")
legend("topright",c('media=0.0006823061',"varianza=1.321808e-06","grado=3"))

####Simulaciones funcion softplus sobre el intervalo [-1,1]
simulations_MSE_softplus_11<-matrix(, nrow = n_simulation, ncol = 3)
for (i in 1:n_simulation ) {
  ejemplo<-ejecutar_ejemplo(n_sample, p, q_original, mean_range, beta_range, error_var, scale_method = "-1,1",h_1, softplus, q_taylor=3, stepmax)
  simulations_MSE_softplus_11[i,1]<-ejemplo$MSE.NN.vs.PC
  simulations_MSE_softplus_11[i,2]<-ejemplo$MSE.NN.VS.PR
  simulations_MSE_softplus_11[i,3]<-ejemplo$MSE.NN.vs.poli_regre
}

#Softplus_11_Chebyshev
print(mean(simulations_MSE_softplus_11[,1]))
print(var(simulations_MSE_softplus_11[,1]))
x11()
boxplot(simulations_MSE_softplus_11[,1], horizontal = TRUE, ylim=c(0,0.1), col = "orange", main="Función de Activación Softplus")
legend("topright",c('media=0.03553674',"varianza=0.06349858","Chebyshev=3"))

#Softplus_11_Taylor
print(mean(simulations_MSE_softplus_11[,2]))
print(var(simulations_MSE_softplus_11[,2]))
x11()
boxplot(simulations_MSE_softplus_11[,2], horizontal = TRUE, ylim=c(0,0.1), col = "tomato", main="Función de Activación Softplus")
legend("topright",c('media=0.04244521',"varianza=0.08100633","Taylor=3"))

#Softplus_11_Regre_poli
print(mean(simulations_MSE_softplus_11[,3]))
print(var(simulations_MSE_softplus_11[,3]))
x11()
boxplot(simulations_MSE_softplus_11[,3], horizontal = TRUE, ylim=c(0,0.01), col = "yellow", main="Función de Activación Softplus")
legend("topright",c('media=0.002884632',"varianza=1.92752e-05","grado=3"))

####Simulaciones funcion sigmoide sobre el intervalo [0,1]
simulations_MSE_sigmoide_01<-matrix(, nrow = n_simulation, ncol = 3)
for (i in 1:n_simulation ) {
  ejemplo<-ejecutar_ejemplo(n_sample, p, q_original, mean_range, beta_range, error_var, scale_method = "0,1",h_1, sigmoide, q_taylor=3, stepmax)
  simulations_MSE_sigmoide_01[i,1]<-ejemplo$MSE.NN.vs.PC
  simulations_MSE_sigmoide_01[i,2]<-ejemplo$MSE.NN.VS.PR
  simulations_MSE_sigmoide_01[i,3]<-ejemplo$MSE.NN.vs.poli_regre
}

#Sigmoide_01_Chebyshev
print(mean(simulations_MSE_sigmoide_01[,1]))
print(var(simulations_MSE_sigmoide_01[,1]))
x11()
boxplot(simulations_MSE_sigmoide_01[,1], horizontal = TRUE, ylim=c(0,1), col = "orange", main="Función de Activación Sigmoide")
legend("topright",c('media=24552050',"varianza=2.233666e+17","Chebyshev=3"))

#Sigmoide_01_Taylor
print(mean(simulations_MSE_sigmoide_01[,2]))
print(var(simulations_MSE_sigmoide_01[,2]))
x11()
boxplot(simulations_MSE_sigmoide_01[,2], horizontal = TRUE, ylim=c(0,1), col = "tomato", main="Función de Activación Sigmoide")
legend("topright",c('media=29771223',"varianza=3.284147e+17","Taylor=3"))

#Sigmoide_01_Regre_poli
print(mean(simulations_MSE_sigmoide_01[,3]))
print(var(simulations_MSE_sigmoide_01[,3]))
x11()
boxplot(simulations_MSE_sigmoide_01[,3], horizontal = TRUE, ylim=c(0,0.01), col = "yellow", main="Función de Activación Sigmoide")
legend("topright",c('media=0.0006720471',"varianza=1.096194e-06","Taylor=3"))

####Simulaciones funcion sigmoide sobre el intervalo [-1,1]
simulations_MSE_sigmoide_11<-matrix(, nrow = n_simulation, ncol = 3)
for (i in 1:n_simulation ) {
  ejemplo<-ejecutar_ejemplo(n_sample, p, q_original, mean_range, beta_range, error_var, scale_method = "-1,1",h_1, sigmoide, q_taylor=3, stepmax)
  simulations_MSE_sigmoide_11[i,1]<-ejemplo$MSE.NN.vs.PC
  simulations_MSE_sigmoide_11[i,2]<-ejemplo$MSE.NN.VS.PR
  simulations_MSE_sigmoide_11[i,3]<-ejemplo$MSE.NN.vs.poli_regre
}

#Sigmoide_11_Chebyshev
print(mean(simulations_MSE_sigmoide_11[,1]))
print(var(simulations_MSE_sigmoide_11[,1]))
x11()
boxplot(simulations_MSE_sigmoide_11[,1], horizontal = TRUE, ylim=c(0,5), col = "orange", main="Función de Activación Sigmoide")
legend("topright",c('media= 2.705064',"varianza=371.5573","Chebyshev=3"))

#Sigmoide_11_Taylor
print(mean(simulations_MSE_sigmoide_11[,2]))
print(var(simulations_MSE_sigmoide_11[,2]))
x11()
boxplot(simulations_MSE_sigmoide_11[,2], horizontal = TRUE, ylim=c(0,5), col = "tomato", main="Función de Activación Sigmoide")
legend("topright",c('media=3.369211',"varianza=565.6408","Taylor=3"))

#Sigmoide_11_Regre_poli
print(mean(simulations_MSE_sigmoide_11[,3]))
print(var(simulations_MSE_sigmoide_11[,3]))
x11()
boxplot(simulations_MSE_sigmoide_11[,3], horizontal = TRUE, ylim=c(0,0.01), col = "yellow", main="Función de Activación Sigmoide")
legend("topright",c('media=0.002853602',"varianza=1.740675e-05","Taylor=3"))

####Simulaciones funcion tanh sobre el intervalo [0,1]
simulations_MSE_tanh_01<-matrix(, nrow = n_simulation, ncol = 3)
for (i in 1:n_simulation ) {
  ejemplo<-ejecutar_ejemplo(n_sample, p, q_original, mean_range, beta_range, error_var, scale_method = "0,1",h_1, mi_tanh, q_taylor=3, stepmax)
  simulations_MSE_tanh_01[i,1]<-ejemplo$MSE.NN.vs.PC
  simulations_MSE_tanh_01[i,2]<-ejemplo$MSE.NN.VS.PR
  simulations_MSE_tanh_01[i,3]<-ejemplo$MSE.NN.vs.poli_regre
}

#Tanh_01_Chebyshev
print(mean(simulations_MSE_tanh_01[,1]))
print(var(simulations_MSE_tanh_01[,1]))
x11()
boxplot(simulations_MSE_tanh_01[,1], horizontal = TRUE, ylim=c(0,80), col = "orange", main="Función de Activación Tanh")
legend("topright",c('media=4877978',"varianza=6.35748e+14","Chebyshev=3"))

#Tanh_01_Taylor
print(mean(simulations_MSE_tanh_01[,2]))
print(var(simulations_MSE_tanh_01[,2]))
x11()
boxplot(simulations_MSE_tanh_01[,2], horizontal = TRUE, ylim=c(0,180), col = "tomato", main="Función de Activación Tanh")
legend("topright",c('media=9834137',"varianza=2.581968e+15","Taylor=3"))

#Tanh_01_Regre_poli
print(mean(simulations_MSE_tanh_01[,3]))
print(var(simulations_MSE_tanh_01[,3]))
x11()
boxplot(simulations_MSE_tanh_01[,3], horizontal = TRUE, ylim=c(0,0.01), col = "yellow", main="Función de Activación Tanh")
legend("topright",c('media=0.0006720471',"varianza=1.096194e-06","Taylor=3"))

####Simulaciones funcion sigmoide sobre el intervalo [-1,1]
simulations_MSE_tanh_11<-matrix(, nrow = n_simulation, ncol = 3)
for (i in 1:n_simulation ) {
  ejemplo<-ejecutar_ejemplo(n_sample, p, q_original, mean_range, beta_range, error_var, scale_method = "-1,1",h_1, mi_tanh, q_taylor=3, stepmax)
  simulations_MSE_tanh_11[i,1]<-ejemplo$MSE.NN.vs.PC
  simulations_MSE_tanh_11[i,2]<-ejemplo$MSE.NN.VS.PR
  simulations_MSE_tanh_11[i,3]<-ejemplo$MSE.NN.vs.poli_regre
}


#Tanh_11_Chebyshev
print(mean(simulations_MSE_tanh_11[,1]))
print(var(simulations_MSE_tanh_11[,1]))
x11()
boxplot(simulations_MSE_tanh_11[,1], horizontal = TRUE, ylim=c(0,5), col = "orange", main="Función de Activación Tanh")
legend("topright",c('media=1659.04',"varianza=1370467421","Chebyshev=3"))

#Tanh_11_Taylor
print(mean(simulations_MSE_tanh_11[,2]))
print(var(simulations_MSE_tanh_11[,2]))
x11()
boxplot(simulations_MSE_tanh_11[,2], horizontal = TRUE, ylim=c(0,5), col = "tomato", main="Función de Activación Tanh")
legend("topright",c('media=3362.31',"varianza=5625122305","Taylor=3"))

#Tanh_11_Regre_poli
print(mean(simulations_MSE_tanh_11[,3]))
print(var(simulations_MSE_tanh_11[,3]))
x11()
boxplot(simulations_MSE_tanh_11[,3], horizontal = TRUE, ylim=c(0,0.01), col = "yellow", main="Función de Activación Tanh")
legend("topright",c('media=0.003070313',"varianza=2.41826e-05","Taylor=3"))

####Ahora vamos a intentar hacer las simulaciones con una función no analitica

noa<-function(x){
  if (x>0)
  {
    exp(-1/x)
  }
  else {
    0
  }
}

norma<-function(x,y,z){
  sqrt(x^2+y^2+z^2)
}

noa2<-function(x,y,z){
  noa(1-norma(x,y,z)*norma(x,y,z))
}

#####Esto es para generar los datos 
generateNormalData_noa <- function(n_sample, p, mean_range,  error_var) {
  
  # Obtain the values for the means unformly distributed in the given range
  mean_values <- runif(p, mean_range[1], mean_range[2])
  
  X <- rmvnorm(n_sample, mean_values) #Para genetar datos de una distribución normal multivariada, el segundo argumento es el vector de medias 
  
  # intialize the response vector
  Y <- rep(0, n_sample)
  # set a counter
  
  
  # loop over all the sample
  for (i in 1:n_sample) {
    Y[i]<-noa2(X[i,1],X[i,2],X[i,3])
  }
  
  # finally we add some normal errors:
  Y <- Y + rnorm(n_sample, 0, error_var)
  
  # Store all as a data frame
  data <- as.data.frame(cbind(X, Y))
  
  # Output includes the data and the original betas to comapre later
  output <- vector(mode = "list", length = 2)
  output[[1]] <- data
  names(output) <- c("data")
  return(output)
}

ejecutar_ejemplo_noa<-function(n_sample, p, mean_range, error_var, scale_method, h_1, fun, q_taylor, stepmax = 1e+05){
  # Generate the data:
  data_generated_noa <- generateNormalData_noa(n_sample, p,  mean_range, error_var)
  data <- data_generated_noa$data
  
  
  #### Scale the data in the desired interval and separate train and test ####
  
  data_scaled <- ScaleData2(data, scale_method)
  
  aux <- divideTrainTest(data_scaled, train_proportion = 0.75)
  
  train <- aux$train
  test <- aux$test
  
  # To use neural net we need to create the formula as follows, Y~. does not work. This includes all the variables X:
  var.names <- names(train)
  formula <- as.formula(paste("Y ~", paste(var.names[!var.names %in% "Y"], collapse = " + ")))
  
  # train the net:
  nn <- neuralnet(formula, data = train, hidden = h_1, linear.output = T, act.fct = fun, stepmax = stepmax)
  
  #Esta es la regresión polinomial
  regre_poli<-lm(Y~V1+ V2+V3+ V1*V2+ V1*V3+ V3*V2 + V1*V2*V3+V1*V1+V2*V2+ V3*V3, data = train)
  predic_regre_poli<-predict(regre_poli,test)
  
  # obtain the weights from the NN model:
  w <- t(nn$weights[[1]][[1]]) # we transpose the matrix to match the desired dimensions
  v <- nn$weights[[1]][[2]]
  
  # Obtain the vector with the derivatives of the activation function up to the given degree:
  g <- rev(taylor(fun, 0, q_taylor))
  
  ###Hiperparametros
  n<-q_taylor ###El grado de aproximación del polinomio
  m<-q_taylor+1 ##El número de nodos o raices
  
  aux1<-seq(1,m,by=1)
  r_k<--cos((2*aux1-1)*pi/(2*m))
  
  ###Generar la matriz de Vandermonde
  unos<-c()
  for (i in 1:(n+1)) {
    unos[i]=1
    
  }
  T<-matrix(,m,n+1)
  T[,1]<-t(unos)
  T[,2]<-t(r_k)
  for (j in 2:n) {
    T[,j+1]= 2*t(r_k)*T[,j]-T[,j-1]
    
  }
  x_min<--1
  x_max<-1
  ####Calcular los coeficientes
  x_k<-scale_up(r_k,x_min,x_max)
  y_k<-fun(x_k)
  alfa<-solve(t(T)%*%T)%*%t(T)%*%y_k
  
  betas1<-coeficientes(w,v,alfa)
  prediccion_poli<-poli_regre(test,betas1)
  prediccion_poli<-t(prediccion_poli)
  
  # Apply the formula
  coeff <- obtainCoeffsFromWeights(w, v, g)
  
  # Obtain the predicted values for the test data with our Polynomial Regression
  n_test <- length(test$Y)
  PR.prediction <- rep(0, n_test)
  
  for (i in 1:n_test) {
    PR.prediction[i] <- evaluatePR(test[i, seq(p)], coeff) ####Este es el metodo del autor 
  }
  
  # Obtain the predicted values with the NN
  NN.prediction <- predict(nn, test)
  
  # MSE:
  
  NN.MSE <- sum((test$Y - NN.prediction)^2) / n_test
  PR.MSE <- sum((test$Y - PR.prediction)^2) / n_test
  PC.MSE <- sum((test$Y - prediccion_poli)^2) / n_test
  poli_regre.MSE<- sum((test$Y - predic_regre_poli)^2) / n_test
  
  # MSE between NN and PR (because PR is actually approximating the NN, not the actual response Y)
  MSE.NN.vs.PR <- sum((NN.prediction - PR.prediction)^2) / n_test
  MSE.NN.vs.PC <- sum((NN.prediction - prediccion_poli)^2) / n_test
  MSE.NN.vs.poli_regre<- sum((NN.prediction - predic_regre_poli)^2) / n_test
  
  
  output <- vector(mode = "list", length = 18)
  output[[1]] <- train
  output[[2]] <- test
  output[[3]] <- g
  output[[4]] <- nn
  output[[5]] <- coeff
  output[[6]] <- betas1
  output[[7]] <- alfa
  output[[8]] <- NN.prediction
  output[[9]] <- PR.prediction
  output[[10]] <- prediccion_poli
  output[[11]] <- NN.MSE
  output[[12]] <- PR.MSE
  output[[13]]<-PC.MSE
  output[[14]]<-poli_regre.MSE
  output[[15]]<-MSE.NN.vs.PR
  output[[16]]<-MSE.NN.vs.PC
  output[[17]]<-predic_regre_poli
  output[[18]]<-MSE.NN.vs.poli_regre
  
  names(output) <- c("train", 
                     "test", 
                     "coeff_taylor_activacion", 
                     "nn", 
                     "coeff_articulo", 
                     "coeff_Cheby",
                     "coeff_Cheby_funcion_activacion",
                     "NN.prediction", 
                     "PR.prediction", 
                     "Cheby.prediction",
                     "NN.MSE", 
                     "PR.MSE", 
                     "Pc.MSE",
                     "Poli.MSE",
                     "MSE.NN.VS.PR", 
                     "MSE.NN.vs.PC",
                     "regresion_polinomial_prediccion",
                     "MSE.NN.vs.poli_regre")
  
  
  return(output)
  
}

####Simulaciones funcion softplus sobre el intervalo [0,1]
simulations_MSE_softplus_01_noa<-matrix(, nrow = n_simulation, ncol = 5)
for (i in 1:n_simulation ) {
  ejemplo<-ejecutar_ejemplo_noa(n_sample, p, mean_range,  error_var, scale_method = "0,1",h_1, softplus, q_taylor=3, stepmax)
  simulations_MSE_softplus_01_noa[i,1]<-ejemplo$MSE.NN.vs.PC
  simulations_MSE_softplus_01_noa[i,2]<-ejemplo$MSE.NN.VS.PR
  simulations_MSE_softplus_01_noa[i,3]<-ejemplo$MSE.NN.vs.poli_regre
  simulations_MSE_softplus_01_noa[i,4]<-ejemplo$Poli.MSE
  simulations_MSE_softplus_01_noa[i,5]<-ejemplo$NN.MSE
}

#Media y varianza
print(mean(simulations_MSE_softplus_01_noa[,1]))
print(mean(simulations_MSE_softplus_01_noa[,2]))
print(mean(simulations_MSE_softplus_01_noa[,3]))
print(var(simulations_MSE_softplus_01_noa[,1]))
print(var(simulations_MSE_softplus_01_noa[,2]))
print(var(simulations_MSE_softplus_01_noa[,3]))

###Esta es una prueba para implimir la red
prueba1<-ejecutar_ejemplo(n_sample,p,q_original,mean_range,beta_range,error_var,'0,1',h_1,sigmoide,q_taylor = 5,stepmax = 1e+05)

plot(prueba1$nn)

prueba1v<-prueba1$nn$weights[[1]][[2]]
prueba1w<-prueba1$nn$weights[[1]][[1]]

####Una nueva funcion para los coeficientes
coefficients2<-function(w,v,alfa){
  aux1_beta0<-v[1]+V[2]*(alfa[1]-alfa[3]+alfa[5])
  aux2_beta0<-V[3]*(alfa[1]-alfa[3]+alfa[5])
  aux3_beta0<-V[4]*(alfa[1]-alfa[3]+alfa[5])
  aux4_beta0<-V[5]*(alfa[1]-alfa[3]+alfa[5])
  aux5_beta0<-v[2]*w[1,1]*(alfa[2]-3*alfa[4]+5*alfa[6]+2*alfa[3]*w[1,1]-8*alfa[5]*w[1,1]+4*alfa[4]*(w[1,1])^2-20*alfa[6]*(w[1,1])^2+8*alfa[5]*(w[1,1])^3+16*alfa[6]*(w[1,1])^4)
  aux6_beta0<-v[3]*w[2,1]*(alfa[2]-3*alfa[4]+5*alfa[6]+2*alfa[3]*w[2,1]-8*alfa[5]*w[2,1]+4*alfa[4]*(w[2,1])^2-20*alfa[6]*(w[2,1])^2+8*alfa[5]*(w[2,1])^3+16*alfa[6]*(w[2,1])^4)
  aux7_beta0<-v[4]*w[3,1]*(alfa[2]-3*alfa[4]+5*alfa[6]+2*alfa[3]*w[3,1]-8*alfa[5]*w[3,1]+4*alfa[4]*(w[3,1])^2-20*alfa[6]*(w[3,1])^2+8*alfa[5]*(w[3,1])^3+16*alfa[6]*(w[3,1])^4)
  aux8_beta0<-v[5]*w[4,1]*(alfa[2]-3*alfa[4]+5*alfa[6]+2*alfa[3]*w[4,1]-8*alfa[5]*w[4,1]+4*alfa[4]*(w[4,1])^2-20*alfa[6]*(w[4,1])^2+8*alfa[5]*(w[4,1])^3+16*alfa[6]*(w[4,1])^4)
  beta0<-aux1_beta0+aux2_beta0+aux3_beta0+aux4_beta0+aux5_beta0+aux6_beta0+aux7_beta0+aux8_beta0
  aux1_beta1<-v[2]*w[1,2]*(alfa[2]-3*alfa[4]+5*alfa[6]+4*alfa[3]*w[1,1]-16*alfa[5]*w[1,1]+12*alfa[4]*(w[1,1])^2-60*alfa[6]*(w[1,1])^2+32*alfa[5]*(w[1,1])^3+80*alfa[6]*(w[1,1])^4)
  aux2_beta1<-v[3]*w[2,2]*(alfa[2]-3*alfa[4]+5*alfa[6]+4*alfa[3]*w[2,1]-16*alfa[5]*w[2,1]+12*alfa[4]*(w[2,1])^2-60*alfa[6]*(w[2,1])^2+32*alfa[5]*(w[2,1])^3+80*alfa[6]*(w[2,1])^4)
  aux3_beta1<-v[4]*w[3,2]*(alfa[2]-3*alfa[4]+5*alfa[6]+4*alfa[3]*w[3,1]-16*alfa[5]*w[3,1]+12*alfa[4]*(w[3,1])^2-60*alfa[6]*(w[3,1])^2+32*alfa[5]*(w[3,1])^3+80*alfa[6]*(w[3,1])^4)
  aux4_beta1<-v[5]*w[4,2]*(alfa[2]-3*alfa[4]+5*alfa[6]+4*alfa[3]*w[4,1]-16*alfa[5]*w[4,1]+12*alfa[4]*(w[4,1])^2-60*alfa[6]*(w[4,1])^2+32*alfa[5]*(w[4,1])^3+80*alfa[6]*(w[4,1])^4)
  beta1<-aux1_beta1+aux2_beta1+aux3_beta1+aux4_beta1
  aux1_beta11<-v[2]*(w[1,2])^2*(2*alfa[3]-8*alfa[5]+12*alfa[4]*w[1,1]-60*alfa[6]*w[1,1]+48*alfa[5]*(w[1,1])^2+160*alfa[6]*(w[1,1])^3)
  aux2_beta11<-v[3]*(w[2,2])^2*(2*alfa[3]-8*alfa[5]+12*alfa[4]*w[2,1]-60*alfa[6]*w[2,1]+48*alfa[5]*(w[2,1])^2+160*alfa[6]*(w[2,1])^3)
  aux3_beta11<-v[4]*(w[3,2])^2*(2*alfa[3]-8*alfa[5]+12*alfa[4]*w[3,1]-60*alfa[6]*w[3,1]+48*alfa[5]*(w[3,1])^2+160*alfa[6]*(w[3,1])^3)
  aux4_beta11<-v[5]*(w[4,2])^2*(2*alfa[3]-8*alfa[5]+12*alfa[4]*w[4,1]-60*alfa[6]*w[4,1]+48*alfa[5]*(w[4,1])^2+160*alfa[6]*(w[4,1])^3)
  beta11<-aux1_beta11+aux2_beta11+aux3_beta11+aux4_beta11
  aux1_beta111<-v[2]*(w[1,2])^3*(4*alfa[4]-20*alfa[6]+32*alfa[5]*w[1,1]+160*alfa[6]*(w[1,1])^2)
  aux2_beta111<-v[3]*(w[2,2])^3*(4*alfa[4]-20*alfa[6]+32*alfa[5]*w[2,1]+160*alfa[6]*(w[2,1])^2)
  aux3_beta111<-v[4]*(w[3,2])^3*(4*alfa[4]-20*alfa[6]+32*alfa[5]*w[3,1]+160*alfa[6]*(w[3,1])^2)
  aux4_beta111<-v[5]*(w[4,2])^3*(4*alfa[4]-20*alfa[6]+32*alfa[5]*w[4,1]+160*alfa[6]*(w[4,1])^2)
  beta111<-aux1_beta111+aux2_beta111+aux3_beta111+aux4_beta111
  aux1_beta1111<-v[2]*(w[1,2])^4*(8*alfa[5]+80*alfa[6]*w[1,1])
  aux2_beta1111<-v[3]*(w[2,2])^4*(8*alfa[5]+80*alfa[6]*w[2,1])
  aux3_beta1111<-v[4]*(w[3,2])^4*(8*alfa[5]+80*alfa[6]*w[3,1])
  aux4_beta1111<-v[5]*(w[4,2])^4*(8*alfa[5]+80*alfa[6]*w[4,1])
  beta1111<-aux1_beta1111+aux2_beta1111+aux3_beta1111+aux4_beta1111
  aux1_beta11111<-16*alfa[6]*(v[2]*(w[1,2])^5+v[3]*(w[2,2])^5+v[4]*(w[3,2])^5+v[5]*(w[4,2])^5)
  aux1_beta2<-v[2]*w[1,3]*(alfa[2]-3*alfa[4]+5*alfa[6]+4*alfa[3]*w[1,1]-16*alfa[5]*w[1,1]+12*alfa[4]*(w[1,1])^2-60*alfa[6]*(w[1,1])^2+32*alfa[5]*(w[1,1])^3+80*alfa[6]*(w[1,1])^4)
  aux2_beta2<-v[3]*w[2,3]*(alfa[2]-3*alfa[4]+5*alfa[6]+4*alfa[3]*w[2,1]-16*alfa[5]*w[2,1]+12*alfa[4]*(w[2,1])^2-60*alfa[6]*(w[2,1])^2+32*alfa[5]*(w[2,1])^3+80*alfa[6]*(w[2,1])^4)
  aux3_beta2<-v[4]*w[3,3]*(alfa[2]-3*alfa[4]+5*alfa[6]+4*alfa[3]*w[3,1]-16*alfa[5]*w[3,1]+12*alfa[4]*(w[3,1])^2-60*alfa[6]*(w[3,1])^2+32*alfa[5]*(w[3,1])^3+80*alfa[6]*(w[3,1])^4)
  aux4_beta2<-v[5]*w[4,3]*(alfa[2]-3*alfa[4]+5*alfa[6]+4*alfa[3]*w[4,1]-16*alfa[5]*w[4,1]+12*alfa[4]*(w[4,1])^2-60*alfa[6]*(w[4,1])^2+32*alfa[5]*(w[4,1])^3+80*alfa[6]*(w[4,1])^4)
  beta2<-aux1_beta2+aux2_beta2+aux3_beta2+aux4_beta2
  aux1_beta12<-v[2]*w[1,2]*w[1,3]*(4*alfa[3]-16*alfa[5]+24*alfa[4]*w[1,1]-120*alfa[6]*w[1,1]+96*alfa[5]*(w[1,1])^2+320*alfa[6]*(w[1,1])^3)
  aux2_beta12<-v[3]*w[2,2]*w[2,3]*(4*alfa[3]-16*alfa[5]+24*alfa[4]*w[2,1]-120*alfa[6]*w[2,1]+96*alfa[5]*(w[2,1])^2+320*alfa[6]*(w[2,1])^3)
  aux3_beta12<-v[4]*w[3,2]*w[3,3]*(4*alfa[3]-16*alfa[5]+24*alfa[4]*w[3,1]-120*alfa[6]*w[3,1]+96*alfa[5]*(w[3,1])^2+320*alfa[6]*(w[3,1])^3)
  aux4_beta12<-v[5]*w[4,2]*w[4,3]*(4*alfa[3]-16*alfa[5]+24*alfa[4]*w[4,1]-120*alfa[6]*w[4,1]+96*alfa[5]*(w[4,1])^2+320*alfa[6]*(w[4,1])^3)
  beta12<-aux1_beta12+aux2_beta12+aux3_beta12+aux4_beta12
  aux1_beta112<-v[2]*(w[1,2])^2*w[1,3]*(12*alfa[4]-60*alfa[6]+96*alfa[5]*w[1,1]+480*alfa[6]*(w[1,1])^2)
  aux2_beta112<-v[3]*(w[2,2])^2*w[2,3]*(12*alfa[4]-60*alfa[6]+96*alfa[5]*w[2,1]+480*alfa[6]*(w[2,1])^2)
  aux3_beta112<-v[4]*(w[3,2])^2*w[3,3]*(12*alfa[4]-60*alfa[6]+96*alfa[5]*w[3,1]+480*alfa[6]*(w[3,1])^2)
  aux4_beta112<-v[5]*(w[4,2])^2*w[4,3]*(12*alfa[4]-60*alfa[6]+96*alfa[5]*w[4,1]+480*alfa[6]*(w[4,1])^2)
  beta112<-aux1_beta112+aux2_beta112+aux3_beta112+aux4_beta112
  aux1_beta1112<-v[2]*(w[1,2])^3*w[1,3]*(32*alfa[5]+320*alfa[6]*w[1,1])
  aux2_beta1112<-v[3]*(w[2,2])^3*w[2,3]*(32*alfa[5]+320*alfa[6]*w[2,1])
  aux3_beta1112<-v[4]*(w[3,2])^3*w[3,3]*(32*alfa[5]+320*alfa[6]*w[3,1])
  aux4_beta1112<-v[5]*(w[4,2])^3*w[4,3]*(32*alfa[5]+320*alfa[6]*w[4,1])
  beta1112<-aux1_beta1112+aux2_beta1112+aux3_beta1112+aux4_beta1112
  beta_11112<-80*alfa[6]*(v[2]*(w[1,2])^4*w[1,3]+v[3]*(w[2,2])^4*w[2,3]+v[4]*(w[3,2])^4*w[3,3]+v[5]*(w[4,2])^4*w[4,3])
  aux1_beta22<-v[2]*(w[1,3])^2*(2*alfa[3]-8*alfa[5]+12*alfa[4]*w[1,1]-60*alfa[6]*w[1,1]+48*alfa[5]*(w[1,1])^2+160*alfa[6]*(w[1,1])^3)
  aux2_beta22<-v[3]*(w[2,3])^2*(2*alfa[3]-8*alfa[5]+12*alfa[4]*w[2,1]-60*alfa[6]*w[2,1]+48*alfa[5]*(w[2,1])^2+160*alfa[6]*(w[2,1])^3)
  aux3_beta22<-v[4]*(w[3,3])^2*(2*alfa[3]-8*alfa[5]+12*alfa[4]*w[3,1]-60*alfa[6]*w[3,1]+48*alfa[5]*(w[3,1])^2+160*alfa[6]*(w[3,1])^3)
  aux4_beta22<-v[5]*(w[4,3])^2*(2*alfa[3]-8*alfa[5]+12*alfa[4]*w[4,1]-60*alfa[6]*w[4,1]+48*alfa[5]*(w[4,1])^2+160*alfa[6]*(w[4,1])^3)
  beta22<-aux1_beta22+aux2_beta22+aux3_beta22+aux4_beta22
  aux1_beta122<-v[2]*w[1,2]*(w[1,3])^2*(12*alfa[4]-60*alfa[6]+96*alfa[5]*w[1,1]+480*alfa[6]*(w[1,1])^2)
  aux2_beta122<-v[3]*w[2,2]*(w[2,3])^2*(12*alfa[4]-60*alfa[6]+96*alfa[5]*w[2,1]+480*alfa[6]*(w[2,1])^2)
  aux3_beta122<-v[4]*w[3,2]*(w[3,3])^2*(12*alfa[4]-60*alfa[6]+96*alfa[5]*w[3,1]+480*alfa[6]*(w[3,1])^2)
  aux4_beta122<-v[5]*w[4,2]*(w[4,3])^2*(12*alfa[4]-60*alfa[6]+96*alfa[5]*w[4,1]+480*alfa[6]*(w[4,1])^2)
  beta122<-aux1_beta122+aux2_beta122+aux3_beta122+aux4_beta122
  aux1_beta1122<-v[2]*(w[1,2])^2*(w[1,3])^2*(48*alfa[5]+480*alfa[6]*w[1,1])
  aux2_beta1122<-v[3]*(w[2,2])^2*(w[2,3])^2*(48*alfa[5]+480*alfa[6]*w[2,1])
  aux3_beta1122<-v[4]*(w[3,2])^2*(w[3,3])^2*(48*alfa[5]+480*alfa[6]*w[3,1])
  aux4_beta1122<-v[5]*(w[4,2])^2*(w[4,3])^2*(48*alfa[5]+480*alfa[6]*w[4,1])
  beta1122<-aux1_beta1122+aux2_beta1122+aux3_beta1122+aux4_beta1122
  beta11122<-160*alfa[6]*(v[2]*(w[1,2])^3*(w[1,3])^2+v[3]*(w[2,2])^3*(w[2,3])^2+v[4]*(w[3,2])^3*(w[3,3])^2+v[5]*(w[4,2])^3*(w[4,3])^2)
  aux1_beta222<-v[2]*(w[1,3])^3*(4*alfa[4]-20*alfa[6]+32*alfa[5]*w[1,1]+160*alfa[6]*(w[1,1])^2)
  aux2_beta222<-v[3]*(w[2,3])^3*(4*alfa[4]-20*alfa[6]+32*alfa[5]*w[2,1]+160*alfa[6]*(w[2,1])^2)
  aux3_beta222<-v[4]*(w[3,3])^3*(4*alfa[4]-20*alfa[6]+32*alfa[5]*w[3,1]+160*alfa[6]*(w[3,1])^2)
  aux4_beta222<-v[5]*(w[4,3])^3*(4*alfa[4]-20*alfa[6]+32*alfa[5]*w[4,1]+160*alfa[6]*(w[4,1])^2)
  beta222<-aux1_beta222+aux2_beta222+aux3_beta222+aux4_beta222
  aux1_beta1222<-v[2]*w[1,2]*(w[1,3])^3*(32*alfa[5]+320*alfa[6]*w[1,1])
  aux2_beta1222<-v[3]*w[2,2]*(w[2,3])^3*(32*alfa[5]+320*alfa[6]*w[2,1])
  aux3_beta1222<-v[4]*w[3,2]*(w[3,3])^3*(32*alfa[5]+320*alfa[6]*w[3,1])
  aux4_beta1222<-v[5]*w[4,2]*(w[4,3])^3*(32*alfa[5]+320*alfa[6]*w[4,1])
  beta1222<-aux1_beta1222+aux2_beta1222+aux3_beta1222+aux4_beta1222
  beta11222<-160*alfa[6]*(v[2]*(w[1,2])^2*(w[1,3])^3+v[3]*(w[2,2])^2*(w[2,3])^3+v[4]*(w[3,2])^2*(w[3,3])^3+v[5]*(w[4,2])^2*(w[4,3])^3)
  aux1_beta2222<-v[2]*(w[1,3])^4*(8*alfa[5]+80*alfa[6]*w[1,1])
  aux2_beta2222<-v[3]*(w[2,3])^4*(8*alfa[5]+80*alfa[6]*w[2,1])
  aux3_beta2222<-v[4]*(w[3,3])^4*(8*alfa[5]+80*alfa[6]*w[3,1])
  aux4_beta2222<-v[5]*(w[4,3])^4*(8*alfa[5]+80*alfa[6]*w[4,1])
  beta2222<-aux1_beta2222+aux2_beta2222+aux3_beta2222+aux4_beta2222
  beta12222<-80*alfa[6]*(v[2]*(w[1,2])*(w[1,3])^4+v[3]*(w[2,2])*(w[2,3])^4+v[4]*(w[3,2])*(w[3,3])^4+v[5]*(w[4,2])*(w[4,3])^4)
  beta22222<-16*alfa[6]*(v[2]*(w[1,3])^5+v[3]*(w[2,3])^5+v[4]*(w[3,3])^5+v[5]*(w[4,3])^5)
  aux1_beta3<-v[2]*w[1,4]*(alfa[2]-3*alfa[4]+5*alfa[6]+4*alfa[3]*w[1,1]-16*alfa[5]*w[1,1]+12*alfa[4]*(w[1,1])^2-60*alfa[6]*(w[1,1])^2+32*alfa[5]*(w[1,1])^3+80*alfa[6]*(w[1,1])^4)
  aux2_beta3<-v[3]*w[2,4]*(alfa[2]-3*alfa[4]+5*alfa[6]+4*alfa[3]*w[2,1]-16*alfa[5]*w[2,1]+12*alfa[4]*(w[2,1])^2-60*alfa[6]*(w[2,1])^2+32*alfa[5]*(w[2,1])^3+80*alfa[6]*(w[2,1])^4)
  aux3_beta3<-v[4]*w[3,4]*(alfa[2]-3*alfa[4]+5*alfa[6]+4*alfa[3]*w[3,1]-16*alfa[5]*w[3,1]+12*alfa[4]*(w[3,1])^2-60*alfa[6]*(w[3,1])^2+32*alfa[5]*(w[3,1])^3+80*alfa[6]*(w[3,1])^4)
  aux4_beta3<-v[5]*w[4,4]*(alfa[2]-3*alfa[4]+5*alfa[6]+4*alfa[3]*w[4,1]-16*alfa[5]*w[4,1]+12*alfa[4]*(w[4,1])^2-60*alfa[6]*(w[4,1])^2+32*alfa[5]*(w[4,1])^3+80*alfa[6]*(w[4,1])^4)
  beta3<-aux1_beta3+aux2_beta3+aux3_beta3+aux4_beta3
  aux1_beta13<-v[2]*w[1,2]*w[1,4]*(4*alfa[3]-16*alfa[5]+24*alfa[4]*w[1,1]-120*alfa[6]*w[1,1]+96*alfa[5]*(w[1,1])^2+320*alfa[6]*(w[1,1])^3)
  aux2_beta13<-v[3]*w[2,2]*w[2,4]*(4*alfa[3]-16*alfa[5]+24*alfa[4]*w[2,1]-120*alfa[6]*w[2,1]+96*alfa[5]*(w[2,1])^2+320*alfa[6]*(w[2,1])^3)
  aux3_beta13<-v[4]*w[3,2]*w[3,4]*(4*alfa[3]-16*alfa[5]+24*alfa[4]*w[3,1]-120*alfa[6]*w[3,1]+96*alfa[5]*(w[3,1])^2+320*alfa[6]*(w[3,1])^3)
  aux4_beta13<-v[5]*w[4,2]*w[4,4]*(4*alfa[3]-16*alfa[5]+24*alfa[4]*w[4,1]-120*alfa[6]*w[4,1]+96*alfa[5]*(w[4,1])^2+320*alfa[6]*(w[4,1])^3)
  beta13<-aux1_beta13+aux2_beta13+aux3_beta13+aux4_beta13
  aux1_beta113<-v[2]*w[1,4]*(w[1,2])^2*(12*alfa[4]-60*alfa[6]+96*alfa[5]*w[1,1]+480*alfa[6]*(w[1,1])^2)
  aux2_beta113<-v[3]*w[2,4]*(w[2,2])^2*(12*alfa[4]-60*alfa[6]+96*alfa[5]*w[2,1]+480*alfa[6]*(w[2,1])^2)
  aux3_beta113<-v[4]*w[3,4]*(w[3,2])^2*(12*alfa[4]-60*alfa[6]+96*alfa[5]*w[3,1]+480*alfa[6]*(w[3,1])^2)
  aux4_beta113<-v[5]*w[4,4]*(w[4,2])^2*(12*alfa[4]-60*alfa[6]+96*alfa[5]*w[4,1]+480*alfa[6]*(w[4,1])^2)
  beta113<-aux1_beta113+aux2_beta113+aux3_beta113+aux4_beta113
  }

