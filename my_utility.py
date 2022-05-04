# My Utility : auxiliars functions

import pandas as pd
import numpy  as np
import copy

# Initialize weights of the Deep-AE
def iniW(input,nodesEnc):
    W = []
    prev = input
    for n in range(len(nodesEnc)):
        W.append(randW(nodesEnc[n],prev))
        prev = nodesEnc[n]
    for n in reversed(W):
        W.append(randW(n.shape[1],n.shape[0]))
    return(W)

# Initialize random weights
def randW(next,prev):
    r  = np.sqrt(6/(next+ prev))
    w  = np.random.rand(next,prev)
    w  = w*2*r-r
    return(w)


#gets miniBatch
def dat_miniBatch(i,x,bsize): #toma un x y retorna un array de bsize y el numero de caracteristicas
    xe = x.T[bsize*i:bsize*(i+1)]
    xe = xe.T
    return(xe)

    return()
# Feed-forward of DAE
def forward_dae(x,w):	
    Act=[]
    a0=x
    
    z=np.dot(w[0],x)
    a1=act_func(z)
    
    Act.append(a0)
    Act.append(a1)
    
    ai=a1
    
    for i in range(len(w)):
        if i != 0:
            zi=np.dot(w[i],ai)
            ai=act_func(zi)
            Act.append(ai)
    return(Act)     


# Encoder
def encoder(x,w):
    for weight in w:
        x = act_func(np.dot(weight,x))
    return(x)


#Activation function
def act_func(z):  #provisional
    return(1/(1+np.exp(-z)))   

    # return (np.maximum(0,z))
    
 
# Derivate of the activation funciton
def deriva_func(a): #provisional
    # x = a.copy()
    # x[a<=0] = 0
    # x[a>0] = 1
    # return x
    return(a*(1-a))


#Forward Softmax
def softmax(z):
        exp_z = np.exp(z-np.max(z))
        return(exp_z/exp_z.sum(axis=0,keepdims=True))
# Métrica
def metricas(x,y):
    ...
    return()    
#Confusuon matrix
def confusion_matrix(x,y):
    ...
    return()
#-------------------------------------------------------------------------

