import numpy as np
import matplotlib.pyplot as plt
import math as m


def norme_carre(x):
    return (np.dot(np.transpose(x),x))

def grad_solve(A,b,x_0,eps=0.1):
    r_0 = b - np.dot(A,x_0)
    p_0 = r_0
    x=np.copy(x_0)    
    k=0
    norme_r_0=norme_carre(r_0)
    while(norme_r_0 > eps ):
        norme_r_0 = norme_carre(r_0)
        alpha = norme_r_0/np.transpose(p_0).dot(A).dot(p_0)
        x_1 = x + alpha * p_0
        r_1 = r_0 - alpha * A.dot(p_0)
        beta = norme_carre(r_1)/norme_r_0
        p_1 =r_1 + beta * p_0
        p_0=p_1
        x=x_1
        r_0=r_1
        k=k+1
    return(x)    

A = np.array([[3,0],[0,5]])
b = np.array([0,1])
x_0=np.array([0,0])
eps=0.1
print(grad_solve(A,b,x_0,eps))



