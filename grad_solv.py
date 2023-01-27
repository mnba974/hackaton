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
    while(norme_r_0 > eps):
        alpha = norme_r_0/(np.transpose(p_0).dot(A).dot(p_0))
        x_1 = x + alpha * p_0
        r_1 = r_0 - alpha * A.dot(p_0)
        beta = norme_carre(r_1)/norme_r_0
        p_1 =r_1 + beta * p_0
        p_0=p_1
        x=x_1
        r_0=r_1
        norme_r_0 = norme_carre(r_0)
        k=k+1
    return(x)    


def biconj_grad_solv(A,b,x_0,eps):
    x=x_0
    r_0 = b - np.dot(A,x)
    r_0_bis = r_0
    ro_0 , alpha , w_0 = 1 , 1 , 1 
    v,p = 0,0
    norm_r0 = np.linalg.norm(r_0)
    while (np.linalg.norm(r_0) > eps):
        ro_1=np.dot(r_0_bis, r_0 )
        beta = (ro_1/ro_0) * (alpha / w_0)
        p = r_0 + beta * (p - w_0 * v)
        v = np.dot(A,p)
        alpha = ro_1/np.dot(r_0_bis, v)
        h = x + alpha*p
        if (np.linalg.norm(b - np.dot(A,h))< eps):
            x=h
            return(x)
        s= r_0 - alpha*v
        t = np.dot(A,s) 
        w_1 = (np.dot(t,s))/(np.dot(t,t)) 
        x = h + w_1*s
        if (np.linalg.norm(b-np.dot(A,x))< eps):
            return(x)
        r_0 = s - w_1 * t 






A=np.array([[1,0,0,0],[0,2,0,0],[0,0,3,0],[0,0,0,4]])
b=np.array([1,1,1,1])
print(biconj_grad_solv(A,b,np.array([0,0,0,0]),0.1))


