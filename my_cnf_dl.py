# My Utility : auxiliars functions of configuration
import pandas as pd
import numpy  as np


# Configuration of Deep Laerning 

def config_dl():      
    par = np.genfromtxt("param_dae.csv",delimiter=',')    
    par_dae=[]    
    par_dae.append(np.int16(par[0])) # maxIter
    par_dae.append(np.int16(par[1])) # miniBatchSize
    par_dae.append(np.float(par[2])) # LearningRate
    for i in range(3,len(par)):
        par_dae.append(np.int16(par[i]))
    par = np.genfromtxt("param_sftm.csv",delimiter=',')
    par_sft= []
    par_sft.append(np.int16(par[0]))   #MaxIters
    par_sft.append(np.float(par[1]))   #Learning 

    return(par_dae,par_sft)
#-------------------------------------------------------------------------
# Load data 
def load_data(fname):
    data = pd.read_csv(fname,header=None)
    xe = data.iloc[:-1, :]
    xe = np.array(xe)
    ye = data.iloc[-1, :]
    ye = pd.get_dummies(ye) #one hot encoder
    ye = np.array(ye)
    ye = ye.T

    return(xe,ye)
# Label binary from raw data 
def Label_binary(y):
    ...
    return(label)
   
#-----------------------------------------------------------------------
# save weights of the DL
def save_w_dl(W,Ws,csv_W,cost,csv_cost):    
    np.savetxt(csv_cost, cost, delimiter=",")
    W.append(Ws)
    np.savez(csv_W, W=W)
    return ()
    
#load weight of the DL in numpy format
def load_w_dl():
    ...
    return(W)    
#-----------------------------------------------------------------------

