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
    costo=[]
    for iter in range(param[0]):
        gradW, cost = opt.grad_sftm(x, y, w)
        costo.append(cost)
        w = w - (param[1]*gradW)
        if iter % 500 == 0:
            print("Costo iteracion "+ str(iter) + " softmax:" + str(cost))
    print("Costo final softmax: "+str(cost))
    return(w,costo)



def get_miniBatch(i,x,bsize): #toma un x y retorna un array de bsize y el numero de caracteristicas
    xe = x.T[bsize*i:bsize*(i+1)]
    xe = xe.T
    return(xe)

# Deep AE's Training 
def train_dae(x,W,numBatch,BatchSize,mu):
    for i in range(1):       
        x_batch = ut.dat_miniBatch(i, x, BatchSize)      
        a  = ut.forward_dae(x_batch,W)              
        gW = opt.grad_dae(a, W)
        W = opt.updW_dae(W, gW, mu)
    return(W)

#Deep Learning: Training 
def train_dl(x,param): ##ENTENDER
    numIter = param[0]
    miniBatchSize = param[1]
    learningRate = param[2]
    W        = ut.iniW(x.shape[0],param[3:])
    numBatch = np.int16(np.floor(x.shape[1]/miniBatchSize))
    tau      = learningRate/numIter    #?
    for i in range(numIter):        
        xe  = x[:,np.random.permutation(x.shape[1])]
        mu  = param[2]/(1+tau*i)     
        W   = train_dae(xe,W,numBatch,miniBatchSize,mu)            
    return(W) 


   
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

