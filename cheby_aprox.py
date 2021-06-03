# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 00:25:37 2021

@author: franc
"""

import numpy as np
from scipy import optimize as opt
from numba import njit


import matplotlib.pyplot as plt

x_min= -1
x_max= 1
x_grid = np.linspace(x_min,x_max,100)

def scale_up(z,x_min,x_max):
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

n = 7 # Este es el orden del polinomio
m = 8 # Este es el número de nodos, siempre tiene que cumplirse que m>n+1

# Nodos de Chebyshev (las raices de los polinomos de Chebyshev, un polinomio de Chebyshev de grado m-1 tiene m raices)
r_k = -np.cos((2*np.arange(1,m+1) - 1) * np.pi / (2*m))

T = np.zeros((m,n+1))

T[:,0] = np.ones((m,1)).T

T[:,1] = r_k.T

for i in range(1,n):
    T[:,i+1] = 2 * r_k * T[:,i] - T[:,i-1]
 
#Definición de la función sigmoide
def sigmoid(x):
    return 1/(1 + np.exp(-x))  

#Definición de la función ReLU 
def ReL(x):
    return (x+abs(x))*(1/2)
#Definición de la función Softplus
def Soft(x):
    return np.log(1+np.exp(x))
#Otra figura
def f1(x):
    return (x+ .1)*.1
###
x_k = scale_up(r_k,x_min,x_max)   
y_k = np.tanh(x_k) 
y_k1 = sigmoid(x_k)
y_k2 = ReL(x_k)
y_k3 = f1(x_k)
y_k4 = Soft(x_k)
alfa =np.linalg.inv(T.T @ T) @ T.T @ y_k4

#Evalua las el polinomio en r_k con coeficientes alfa_k
T @ alfa

# se calcula la aproximación de f sobre el intervalo
T = np.zeros((len(x_grid),n+1))

T[:,0] = np.ones((len(x_grid),1)).T

z_grid = scale_down(x_grid,x_min,x_max)

T[:,1] = z_grid.T

for i in range(1,n):
    T[:,i+1] = 2 * z_grid * T[:,i] - T[:,i-1]

# Calcula la aproximación
Tf = T @ alfa

# Estas son las graficas
plt.figure()
plt.plot(x_grid, Soft(x_grid), label="Soft")
plt.plot(x_grid, Tf, label= "Aproximación de orden 3")
plt.title("(x+0.1)0.1")
plt.legend()
plt.show()

# Diferencia
plt.plot(x_grid, Tf-Soft(x_grid), label= "Aproximación de orden 3" )




