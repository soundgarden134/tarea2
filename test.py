
import numpy      as np
import my_cnf_dl  as cnf
import my_utility as ut


# Feed-forward of the DL
def forward_dl(xv,W): 
    L=len(W)
    x=xv
    for i in range(L-1):
        x = ut.act_func(np.dot(W[i],x))
    zv = ut.softmax(np.dot(W[L-1],x))
    return(zv)

# Beginning ...
def main():			
    xv, yv = cnf.load_data('dtest.csv')
    W = cnf.load_w_dl('w_dl.npz')
    zv = forward_dl(xv,W)
    cm,Fsc = ut.metricas(yv,zv) 		
    print('Fsc-mean {:.5f}'.format(Fsc.mean()))
	

if __name__ == '__main__':   
	 main()

