# -*- coding: utf-8 -*-
"""
Created on Sun May 30 11:24:11 2021

@author: franc
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import collections
import pandas as pd
import scipy
import scipy.stats
import seaborn as sns
import statsmodels.api as sm
from scipy.integrate import odeint
from statsmodels.graphics.api import abline_plot

#Este programa es para el primero de la tarea
datos=np.array([7.7, 1.2, 5.5, 5.5, 13.0, 12.0, 9.7, 11.0, 18.0,
19.0, 16.0, 9.3, 6.8, 3.7, 4.8, 6.1, 2.2, 3.6, 9.9, 8.0, 19.0, 11.0, 15.0, 8.1, 15.0, 5.1, 12.0,
7.3, 2.1, 14.0])
datos=datos.tolist()

n=len(datos)
media=np.mean(datos)
varianza=np.var(datos)

#Vamos a sacar el valor más grande de los datos para poner la indicadora de ese valor en la verosimilitud en lugar en del producto de las indicadoras
indice_maximo=datos.index(max(datos))
maximo=datos[indice_maximo]

def pareto(theta,b,c):
    return ((c)**b)*b/(theta**(b+1))

def verosimilitud(theta):
    if(maximo<=theta): return (1/theta**n)
    else: return 0
    
###para escoger los parámetros de la a priori nos fijamos en la media y varianza de los datos que tenemos
def parametro1(x,c):
    return x/(x-c)


aux1=-2*((media**2)+varianza)
aux2=varianza*media+(media**3)   
###Estos son los parametros para la priori
#c=(-aux1-math.sqrt((aux1**2)-4*media*aux2))/(2*media)
#b=parametro1(media,c)

c=10
b=2*media/(2*media-c)


def posterior(theta):
    return pareto(theta,b,c)*verosimilitud(theta)

print(posterior(1))

t2=np.linspace(1/10000,50,10000)
y=[None]*len(t2)
for i in range(len(t2)):
    y[i]=posterior(t2[i])

integral=(t2[1]-t2[0])*np.sum(y)
 
###Esta sería la verdadera posterior   
def posterior2(theta):
    return posterior(theta)/integral

###Esto es para la grafica de esa posterior 
y2=[None]*len(t2)
for i in range(len(t2)):
    y2[i]=posterior2(t2[i])
    
plt.title('Distribución Posterior')
plt.xlabel(r'$\theta$')
plt.plot(t2,y2)
plt.show()


####Ahora hay que calcular la probabibilidades de que theta sea menor o mayor igual que 20

print(max(y2))
indice_max=y2.index(max(y2))
MAP=t2[indice_max] 
print(t2[indice_max]) ###Este seria el map

####Este es el último donde la posterior es menor a 20, es decir, hay que integrar hasta este indice
print(t2[3999])
H0_aux=np.zeros(len(t2))
for i in range(3999):
    H0_aux[i]=y2[i]

###Esta seria la probabilidad de que fuese menor que 20
H0=np.sum(H0_aux)*(t2[1]-t2[0])    

H1_aux=np.zeros(len(t2))
for i in range(4000,len(t2)):
    H1_aux[i]=y2[i]

H1=np.sum(H1_aux)*(t2[1]-t2[0])  

