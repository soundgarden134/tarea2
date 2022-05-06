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


    return np.maximum(0,z)
    
 
# Derivate of the activation funciton
def deriva_func(a): #provisional
    return (a>0).astype(float)



#Forward Softmax
def softmax(z):
        exp_z = np.exp(z-np.max(z))
        return(exp_z/exp_z.sum(axis=0,keepdims=True))
# Métrica
def metricas(x,y):
    confusion_matrix = np.zeros((y.shape[0], x.shape[0]))
    
    for real, predicted in zip(y.T, x.T):
        confusion_matrix[np.argmax(real)][np.argmax(predicted)] += 1
        
    f_score = []
    
    for index, caracteristica in enumerate(confusion_matrix):
        
        TP = caracteristica[index]
        FP = confusion_matrix.sum(axis=0)[index] - TP
        FN = confusion_matrix.sum(axis=1)[index] - TP
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        f_score.append(2 * (precision * recall) / (precision + recall))
        
    metrics = pd.DataFrame(f_score)
    metrics.to_csv("metrica_dl.csv", index=False, header=False)
    f_score = np.array(f_score)
    return(confusion_matrix, f_score) 
#Confusuon matrix
def confusion_matrix(x,y):
    ...
    return()
#-------------------------------------------------------------------------

