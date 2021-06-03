# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 23:07:15 2021

@author: franc
"""

####Este es un ejemplo para predecir si unas estrellas son pulsares o no
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import scipy.optimize as optimize

dataset = pd.read_csv('C:\\Users\\franc\\Downloads\\Datos\\HTRU_2.csv')
X_data = dataset.iloc[:, 0:-1].values #Esta es la matriz que contiene las covariables
Y_data = dataset.iloc[:,-1].values #Esta es la matriz que contiene lo que se quiere predecir

scaler = StandardScaler() ###Esto solo es para estándarizar los datos
X_data = scaler.fit_transform(X_data)

x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data) #Esta función por deafault parte los datos en 75 por ciento para entrenar y el resto para validar
 #Ahora a definir el modelo. La red tiene dos capas una capa oculta con 6 neuronas y una función de activación Relu y la capa de salida que es una función sigmoid

model = keras.Sequential()
model.add(keras.layers.Dense(6, input_shape=(8,), activation='relu',)) 
model.add(keras.layers.Dense(1,activation='sigmoid'))

#Esto es para obtener los pesos
for layer in model.layers:
    weights = layer.get_weights() # list of numpy arrays

#La función de perdida es una función Binary_entropy y el optimizador es Adam
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

#Esto es para ajustar el modelo son 200 épocas con mini lotes de 100
#Ahora vamos a hacer unas modificaciones para que el modelo deje de entrenarse cuando la precisión ya no mejora
early = keras.callbacks.EarlyStopping(monitor='loss', patience=10) #patience es el número de epocas que hay que esperar para que el modelo mejore antes de que se detenga 
history = model.fit(x_train, y_train, epochs = 200 , batch_size=100,  validation_data=(x_test,y_test), callbacks=[early])
## Esto es para imprimir la preción que va teniendo en modelo sobre cada subcnjunto de datos X_train, X_test por época
print(history.history['accuracy'])
print(history.history['val_accuracy'])
#Esta gráfica es para comparar la precisión del modelo sobre X_train y sobre x_test por epóca sive para ver si hay un sobre ajuste o si es necesario entrenar el modelo más tiempo
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#Esta gráfica es para ver como se comporta respecto a la función de perdida 
#Esto es para evaluar el modelo
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#Esto solo es para resumir, te imprime la precisión y te la perdida al final del entrenamiento
model.evaluate(x_test, y_test)

#Esto solo es para ver cuanstas épocas corrió
print(len(history.history['loss']))

#Ahora vamos con la aproximación

#Definición de la función Relu 
def Relu(z):
    return (abs(z)+z)/2

#Definición de la función sigmoide
def sigmoid(x):
    return 1/(1 + np.exp(-x))    

#Esta es la composición
def compo(z):
    return sigmoid(Relu(z))

###Estos son los extremos del intervalo donde se estima que esta el logit
x_min= -5
x_max= 5
x_grid = np.linspace(x_min,x_max,100)

def scale_up(z,x_min,x_max): #estas funciones son solo para reescalar 
    """
    Scales up z \in [-1,1] to x \in [x_min,x_max]
    where z = (2 * (x - x_min) / (x_max - x_min)) - 1
    """
    
    return x_min + (z + 1) * (x_max - x_min) / 2


def scale_down(x,x_min,x_max):
    """
    Scales down x \in [x_min,x_max] to z \in [-1,1]
    where z = f(x) = (2 * (x - x_min) / (x_max - x_min)) - 1
    """    
    
    return (2 * (x - x_min) / (x_max - x_min)) - 1

n = 20 # Este es el orden del polinomio
m = 21  # Este es el número de nodos, siempre tiene que cumplirse que m>n+1

# Nodos de Chebyshev (las raices de los polinomos de Chebyshev, un polinomio de Chebyshev de grado m-1 tiene m raices)
r_k = -np.cos((2*np.arange(1,m+1) - 1) * np.pi / (2*m))

#Esta será la matriz donde cada columna vienen evaluados los polinomios de Cheby en las raices 
T = np.zeros((m,n+1))

T[:,0] = np.ones((m,1)).T

T[:,1] = r_k.T

for i in range(1,n):
    T[:,i+1] = 2 * r_k * T[:,i] - T[:,i-1]

#Esta manda las racices de los polinomios del [-1,1] al [x_min, x_max]
x_k = scale_up(r_k,x_min,x_max)
y_k = compo(x_k) 
alfa =np.linalg.inv(T.T @ T) @ T.T @ y_k #####Alfa son los coeficientes de la combinación lineal de los polinomios de Cheby 

# Use coefficients to compute an approximation of $f(x)$ over the grid of $x$:
T = np.zeros((len(x_grid),n+1))

T[:,0] = np.ones((len(x_grid),1)).T

z_grid = scale_down(x_grid,x_min,x_max)

T[:,1] = z_grid.T

for i in range(1,n):
    T[:,i+1] = 2 * z_grid * T[:,i] - T[:,i-1]

# compute approximation
Tf = T @ alfa

##Esta grafica es solo para asegurar que todo se hizo bien
plt.figure()
plt.plot(x_grid,compo(x_grid))
plt.plot(x_grid, Tf)
plt.show()

#Las siqguientes son funciones auxiliares

def fun1(w1,w2,w3,w4,w5,w6,w7,w8):
   w=np.array([w1,w2,w3,w4,w5,w6,w7,w8])
   return x_train.dot(w)

#Esto es solo para probar que todo va bien
prueba1=fun1(1, 0,0,0,0,0,0,0)

def fun2(z):
    logit = np.zeros((1, n+1))
    logit[0,0]=1
    logit[0,1]=scale_down(z, x_min, x_max)
    logit_scale=scale_down(z, x_min, x_max)
    for i in range(1,n):
        logit[0,i+1]=2*logit_scale*logit[0,i]-logit[0,i-1]
    return logit @ alfa

#Esto es solo para probar que todo va bien
prueba2 = fun2(0)

def fun3(w1,w2,w3,w4,w5,w6,w7,w8):
    w=np.zeros((len(x_train),1))
    w[0,0]=fun2(fun1(w1,w2,w3,w4,w5,w6,w7,w8)[0])
    for i in range(1, len(x_train)):
        w[i,0]=fun2(fun1(w1,w2,w3,w4,w5,w6,w7,w8)[i])
    return w.reshape((len(x_train,)))

#Esto es solo para probar que todo va bien
prueba3 = fun3(1,0,0,0,0,0,0,0)
prueba3_1 = np.multiply(y_train, np.log(prueba3))
prueba3_2= np.ones(len(x_train))-y_train


def fun4(w1,w2,w3,w4,w5,w6,w7,w8):
    aux1= np.multiply(y_train, np.log(fun3(w1,w2,w3,w4,w5,w6,w7,w8)))
    aux2= np.ones(len(x_train))-y_train
    aux3= np.ones(len(x_train))-fun3(w1,w2,w3,w4,w5,w6,w7,w8)
    aux4=np.multiply(aux2, np.log(aux3))
    aux5=aux1+aux4
    return (-1/len(x_train))*sum(aux5)

prueba4 = fun4(1,0,0,0,0,0,0,0)

####Esto es para encontrar los parámetros que minimizan la función
##### Esto es para optimizar
def f(params):
    # print(params)  # <-- you'll see that params is a NumPy array
    w1, w2, w3, w4, w5, w6, w7, w8  = params # <-- for readability you may wish to assign names to the component variables
    return fun4(w1,w2,w3,w4,w5,w6,w7,w8)

initial_guess = [1, 0,0,0,0,0,0,0]
result = optimize.minimize(f, initial_guess)
if result.success:
    fitted_params = result.x
    print(fitted_params)
else:
    raise ValueError(result.message)



    


        
    