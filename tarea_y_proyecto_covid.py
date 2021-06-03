# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:57:01 2021

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
import pytwalk as twalk
from statsmodels.graphics.api import abline_plot

###Este archico es para tratar de resolver el último ejercico de la tarea de bayesiana y también para 
t=np.linspace(0,10, 26)
X0=100
theta1=1
theta2=1000
sigma=30

####Ahora vamos a programar el modelo 
def I(theta1, theta2,t):
    return theta2*X0/((theta2-X0)*np.exp(-theta1*t)+X0)

###Ahora vamos a simular los datos 
normales=np.random.normal(0,sigma,len(t))
datos1=I(theta1,theta2,t)
datos2=datos1+normales

###Ahora vamos a graficar los datos para ver que todo esta en orden
plt.title('Datos simulados')
plt.ylabel('Infectados')
plt.xlabel('Tiempo')
plt.plot(t,datos2,'ro')
plt.show()

###Ahora vamos a programar la logverosimilitud
def logverosimilitud(theta,y=datos2):
    n=len(t)
    s1=n*np.log(np.sqrt(2*math.pi)*sigma)
    s2=(1/2*(sigma)**2)*np.sum((datos1-datos2)**2)
    return -s1-s2


####Ahora vamos a definir la logprior esta es una gamma
#def logprior(theta,alfa=100,beta=100):
 #   s1=alfa*np.log(beta)+(alfa-1)*np.log(theta)
  #  s2=np.log(math.gamma(alfa))+beta*theta
   # return s1-s2

#####Ahora vamos a definir una log prior poco informativa como una Cauchy
def logprior(theta,gama=10,x0=0):
    if (theta<0): return 0
    else: return np.log(2)-np.log(math.pi*gama)-np.log(1+((theta-x0)/gama)**2)


###Ahora vamos a definir la energia 
def energia(theta):
    return -logprior(theta)-logverosimilitud(theta)

###Ahora vamos a definir unos vectores que nos ayudaran a ver los valores de la energía
t2=np.linspace(0,6,10000)
vector1=[None]*len(t2)
for i in range(len(t2)):
    vector1[i]=energia(t2[i])
    
###Ahora vamos por la posterior
def posterior(theta):
    return np.exp(-energia(theta)+7913160)

###Ahora vamos a ver como la distribución posterior
vector2=[None]*len(t2)
for i in range(len(t2)):
    vector2[i]=posterior(t2[i])

###Esta es una grafica de la distribucion posterior
plt.plot(t2,vector2)

####Ahora para que de verdad sea trate de una distribución posterior hay que normalizar para que su integral valga 1
###Vamos a utilizar una suma de Riemann para hacer la integración

integral=np.sum(vector2)*(t2[1]-t2[0])

def posterior2(theta):
    return posterior(theta)*(1/integral)

###Ahora vamos a graficar en detalle la verdadera distrubción posterior
vector3=[None]*len(t2)
for i in range(len(t2)):
    vector3[i]=posterior2(t2[i])

###Esta deberia ser la distribución posterior para theta1
plt.title('Distribución Posterior')
plt.xlabel(r'$\theta_1$')
plt.plot(t2,vector3)
plt.show()

###Veamos que tan bueno es la estimación usando el MAP de la posterior
indice_MAP=vector3.index(max(vector3))

####Este debería ser el MAP
print(t2[indice_MAP])

###Ahora vamos transformar nuestra distribución posterior a un dataframe para que podamos calcular con facilidad sus estadisticas
posterior_theta1=pd.DataFrame(vector3,columns=['valores de la posterior'])


###Esta es la grafica de la posterior
plt.title('Comparación de la estimación')
plt.xlabel('Tiempo')
plt.ylabel('Infectados')
plt.plot(t,datos2,'ro', label='Datos')
plt.plot(t,I(0.990009900099001,theta2,t), '-b' ,label='Estimación')
plt.legend()
plt.show()

###Ahora vamos intentar implementar el T-walk para estimar ambos parametros, pero primero vamos a hacer unas modificaciones a logverosimilitud y a la logprior
def logverosimilitud2(theta):
    theta1, theta2=theta 
    datos1=I(theta1,theta2,t)
    n=len(t)
    s1=n*np.log(np.sqrt(2*math.pi)*sigma)
    s2=(1/2*(sigma)**2)*np.sum((datos1-datos2)**2)
    return -s1-s2

###Ahora vamos a definir la logprior
def logprior2(theta):
    theta1, theta2 =theta
    alfa1=100
    beta1=100
    s1=alfa1*np.log(beta1)+(alfa1-1)*np.log(theta1)
    s2=np.log(math.gamma(alfa1))+beta1*theta1
    k=100
    teta=10
    s3=(k-1)*np.log(theta2)
    s4=(theta2/teta)+np.log(math.gamma(k))+k*np.log(teta)
    return s1-s2+s3-s4

####Ahora vamos a definir la nueva energia
def energia2(theta):
    return -logverosimilitud2(theta)-logprior2(theta)

#####Ahora vamos a definir el punto inicial
def p0():
    theta1=np.random.uniform(0,5,1)
    theta2=np.random.uniform(900,1200,1)
    return np.array([theta1[0],theta2[0]])

###Ahora vamos a definir el soporte
def supp(theta): #funcion de soporte , el t-walk lo va proponiendo
  theta1, theta2=theta
  if(theta1>0 and theta2>0):
    return True
  else:
    return False

###Número de iteraciones
T=200000

####Estos van a ser los puntos iniciales
x0=p0() #punto inicial 1
xp0=p0() #punto inicial 2

tchain = twalk.pytwalk( n=2, U=energia2, Supp=supp ) #n es la dimension y U es la energia
tchain.Run( T=T , x0= x0 , xp0= xp0)

tchain.Ana()
tchain.IAT() #tiempo de autocorrelacion integrado
#que tan rapido mi cadena produce obs independientes?
#Tamano de muestra efectivo

iat=20
print('Tamano de muestra efectivo: ', T/ iat ) #cuantas de tus obs son independientes

####Ahora si vamos a ver los resultados
toutput=tchain.Output[:, 0:2 ]

theta1_est=toutput[:,0]
theta2_est=toutput[:,1]

#burnin
bi=int(0.20*T)
theta1_est=toutput[bi:,0]
theta2_est=toutput[bi:,1]

resultados={'theta1': theta1_est, 'theta2':theta2_est }
resultados=pd.DataFrame(resultados)

##Estas son las estadisticas descriptivas de la distribucipon posterior de cada parametro
pd.DataFrame.describe(resultados)

###Vamos a utilizar los MAP como estimadores
plt.title('Comparación con la estimación')
plt.xlabel('Tiempo')
plt.ylabel('Infectados')
plt.plot(t,datos2, 'ro', label='Datos')
plt.plot(t,I(1.013789,993.936409,t), '-b', label='Curva usando los estimadores')
plt.legend()

##Estos son para las trazas y los histogramas
def histmh(x, variable,  flag=False): #funcion para ver histogramas 
  sns.displot(x, kde=flag, bins=20)  
  plt.xlabel('x')
  plt.title('Histograma de '+ variable)
  
def tplot(x, variable ): #funcion para ver trazas
  plt.plot(x,)
  plt.xlabel('t')
  plt.ylabel( variable)
  plt.title('Trazas')  
  
tplot(theta1_est, r'$\theta_1$'  )
tplot(theta2_est, r'$\theta_2$'  )  

histmh(theta1_est, r'$\theta_1$'  )
histmh(theta2_est, r'$\theta_2$'  )

###Este es para obtener los IBC
intervalos_theta1=scipy.stats.bayes_mvs(resultados['theta1'], alpha=0.95)
intervalos_theta2=scipy.stats.bayes_mvs(resultados['theta2'], alpha=0.95)

print(intervalos_theta1)
print(intervalos_theta2)


#####Aqui viene la parte del proyecto, los casos acumulados de COVID en la CDMX
datos_pro=np.array([ 2., 3., 3., 3., 3., 3., 5., 6., 6., 6.,9., 10., 15., 23., 32., 33., 36., 43., 47.,61., 80., 84., 90., 110., 131., 179., 264.,317., 343., 385., 455., 500., 563., 651.,766., 872., 1045., 1211., 1310., 1512., 1617., 1853., 1997.])
#Esta es la incidencia
y=np.diff(datos_pro)

##Ahora vamos a poner los dias
dias=np.linspace(1, 43, 43)

##Vamos a poner el número de susceptibles
N=8855000 #Este es el número de habitantes en la ciudad de méxico

##Ahora vamos a programar de nuevo I
def I2(I0,beta):
    return N*I0/((N-I0)*np.exp(-beta*dias)+ I0)

####Ahora hay que programar la logverosimilitud
def logL(theta):
    I0, beta = theta
    I=I2(I0,beta)
    L=np.diff(I)
    s1=-np.sum(L)
    s2=np.sum(y*np.log(L))
    return s1+s2

###Ahora vamos a proponer una priori
def logP(theta):
    I0, beta = theta
    gamma=20
    x0=0
    log_beta =scipy.stats.lognorm.logpdf(beta,0.5,loc=math.log(0.4))
    #alfa1=20
    #beta1=10
    #log_beta=alfa1*np.log(beta1)+(alfa1-1)*np.log(beta)-np.log(math.gamma(alfa1))-beta1*beta
    log_I0 = -np.log(math.pi*gamma)-np.log(1+(((I0-x0)/gamma)**2))
    return log_I0+log_beta

###Ahora, esta será la energia
def U(theta):
    return -logP(theta)-logL(theta)

###Estos son los puntos iniciales
def p02():
    I0=np.random.uniform(0,10,1)
    beta=np.random.uniform(0,5,1)
    return np.array([I0[0],beta[0]])

###Este es el soporte
def supp2(theta): #funcion de soporte , el t-walk lo va proponiendo
  I0, beta =theta
  if(I0>0 and beta>0):
    return True
  else:
    return False

###Número de iteraciones
T=200000

####Estos van a ser los puntos iniciales
x02=p02() #punto inicial 1
xp02=p02() #punto inicial 2

tchain2 = twalk.pytwalk( n=2, U=U, Supp=supp2 ) #n es la dimension y U es la energia
tchain2.Run( T=T , x0= x02, xp0= xp02)

tchain2.Ana()
tchain2.IAT() #tiempo de autocorrelacion integrado
#que tan rapido mi cadena produce obs independientes?
#Tamano de muestra efectivo

iat2=44.9

print('Tamano de muestra efectivo: ', T/ iat2 )

####Ahora si vamos a ver los resultados
toutput2=tchain2.Output[:, 0:2 ]

I0_est=toutput2[:,0]
beta_est=toutput2[:,1]

#burnin
bi=int(0.20*T)
I0_est=toutput2[bi:,0]
beta_est=toutput2[bi:,1]

resultados2={'I0': I0_est, 'beta':beta_est }
resultados2=pd.DataFrame(resultados2)

###Estos son los histogramas 
histmh(I0_est, r'$I_0$'  )
histmh(beta_est, r'$\beta$'  )

####Estas son las trazas de los parametros
tplot(I0_est, r'$I_0$'  )
tplot(beta_est, r'$\beta$'  )

medianas=pd.DataFrame.median(resultados2)
print(medianas)
##Estas son las estadisticas descriptivas de la distribucipon posterior de cada parametro
pd.DataFrame.describe(resultados2)

y_aux=np.append(0,y)
esti_aux=I2(5.901746, 0.113007)
esti_aux2=np.append(0,np.diff(esti_aux))
##Vamos a ver si la curva se ajusta bien
plt.title('Comparación para la predicción usando los mínimos')
plt.xlabel('dias')
plt.ylabel('registros diarios')
plt.plot(dias, datos_pro, 'ro', color='orange', label='datos')
plt.plot(dias,esti_aux, label='curva estimada')
plt.legend()
####Ahora con la incidencia
plt.title('Incidencia por día usando los mínimos')
plt.xlabel('dias')
plt.ylabel('nuevos casos')
plt.plot(dias,y_aux,'ro',color='orange',label='y')
plt.plot(dias, esti_aux2,label='curva estimada')
plt.legend()