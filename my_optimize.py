# My Utility : Algorith of optimization for Deep Learning 

import pandas as pd
import numpy  as np
import my_utility as ut
import copy
#-----------------------------------------------------------------------
# STEP: Feed-Backward

def grad_dae(a,w):
    gradW = [None]*len(w)
    deltas = [None]*len(w)
    
    for idx in reversed(range(len(w))):
        if(idx != (len(w)-1)):
            delta_next = deltas[idx+1]
            
            delta_ = np.dot(w[idx+1].T, delta_next)
            da = ut.deriva_func(a[idx+1])
            
            deltaH = delta_ * da
            
            grad = np.dot(deltaH,a[idx].T)
            
            gradW[idx] = grad
            deltas[idx] = deltaH
        else:
            e= a[-1]-a[0]
            da = ut.deriva_func(a[-1])
            
            delta_f = e*da
            
            grad = np.dot(delta_f,a[-2].T)
            
            gradW[-1] = grad
            deltas[-1] = delta_f
    

    return(gradW)  

def grad_dae2(a,w):
    E = 1
    gradW = []
    

    return(gradW)  

# Update DAE's Weight 
def updW_dae2(w,gradW,mu, beta, v):
    for i in range(len(w)):
        w[i] = w[i] + beta*v[i] - mu*gradW[i]
        v[i] = beta*v[i] - mu*gradW[i]
    return(w, v)

def updW_dae(w,gradW,mu):
    for i in range(len(w)):
        tau = mu/len(w)
        mu_k = mu/(1+np.dot(tau,(i+1)))
        w[i] = w[i] - mu_k*gradW[i]
    return(w)

# Softmax's gradient
def grad_sftm(x,y,w):
    z = np.dot(w,x)
    a = ut.softmax(z)
    z = np.array(z, dtype=np.float64)
    a = np.array(a, dtype = np.float64)
    
    ya = y*np.log(a)
    cost = (-1/x.shape[1])*np.sum(np.sum(ya))
    gW = ((-1/x.shape[1])*np.dot((y-a),x.T))
    return(gW,cost)
  

# Update Softmax's Weight 
def updW_sftm(w,v,gW,mu): 
    beta = 0.9
    epsilon = 1e-8
    v = v*beta + (1-beta)*gW**2
    gRMS = (mu/(np.sqrt(v) + epsilon))*gW
    w = w - gRMS
       
    
    return(w, v)
#

