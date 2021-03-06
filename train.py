# Deep-Learning: Training 

import pandas       as pd
import numpy        as np
import my_cnf_dl    as cnf
import my_utility   as ut
import my_optimize  as opt
import copy
	
# Softmax's training
def train_softmax(x,y,param):
    w=ut.randW(y.shape[0],x.shape[0])
    cost_array=[]
    mu = param[1] #learning Rate
    v = 0
    for iter in range(param[0]):
        gradW, cost = opt.grad_sftm(x, y, w)
        cost_array.append(cost)
        w,v = opt.updW_sftm(w, v, gradW, mu)
        if iter % 500 == 0:
            print("Costo iteracion "+ str(iter) + " softmax:" + str(cost))
    print("Costo final softmax: "+str(cost))
    return(w,cost_array)



def get_miniBatch(i,x,bsize): #toma un x y retorna un array de bsize y el numero de caracteristicas
    xe = x.T[bsize*i:bsize*(i+1)]
    xe = xe.T
    return(xe)


def initializeV(W): #inicializa matriz de dimension W
    v = [None]*len(W)
    for i in range(len(W)):
        dim1, dim2 = W[i].shape
        v[i] = np.zeros((dim1,dim2))
    return v


def decoder_weights(W):
    cop = copy.deepcopy(W)
    half = int(len(cop)/2)
    aux = cop[half:]
    aux.reverse()
    for i in range(len(aux)):
        aux[i] = aux[i].T
    return aux
    
    
# Deep AE's Training        
def train_dae(x,W,V,mu,numBatch,BatchSize):
    b = 0.9
    for i in range(numBatch):
        tau = 1-i/numBatch
        beta = tau/((1-b) + (b*tau))
        xe   = get_miniBatch(i,x,BatchSize)
        a  = ut.forward_dae(xe,W)
        gW = opt.grad_dae(a, W)
        W,V  = opt.updW_dae(W,V,gW,mu,beta)
    return(W)


#Deep Learning: Training 
def train_dl(x,param): 
    numIter = param[0]
    BatchSize = param[1]
    learningRate = param[2]
    W        = ut.iniW(x.shape[0],param[3:])
    numBatch = np.int16(np.floor(x.shape[1]/BatchSize))
    tau      = learningRate/numIter  
    V = initializeV(W)
    for i in range(numIter):      
        xe  = x[:,np.random.permutation(x.shape[1])] 
        W = train_dae(xe,W,V,learningRate,numBatch,BatchSize)   
    aux = decoder_weights(W)         
    return(aux) 



# Beginning ...
def main():
    p_dae,p_sftm = cnf.config_dl() 
    xe,ye = cnf.load_data('dtrain.csv')       
    W = train_dl(xe,p_dae) 
    Xr = ut.encoder(xe,W)
    Ws, cost = train_softmax(Xr,ye,p_sftm)
    cnf.save_w_dl(W,Ws,'w_dl.npz',cost,'costo_sftm.csv')

       
if __name__ == '__main__':   
	 main()

