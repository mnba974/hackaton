import numpy as np
import matplotlib.pyplot as plt
import dill
import math as m

delx=0.01
dely=0.01
delt=0.001
Nx,Ny=int(1/delx),int(1/dely)


X=np.linspace(0,1,int(1/delx ))
Y=np.linspace(0,1,int(1/dely ))

def f(x,y):
    return 0.5 +np.sin(2*np.pi*x)*np.sin(2*np.pi*y) -0.5*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)
T=[]
for x in X:
    for y in Y:
        T.append(f(x,y))

T=np.array(T)

MatT=np.array([T])

K=np.zeros((Nx*Ny,Nx*Ny))
for pos in range(Nx*Ny):

    i = pos//Nx
    j = pos%Nx

    if i<Ny-1:
        K[i*Nx+j,(i+1)*Nx+j]+=1/delx**2

    K[i*Nx+j,i*Nx+j]+=-2/delx**2

    if i>0:
        K[i*Nx+j,(i-1)*Nx+j]+=1/delx**2

    if j<Nx-1:
        K[i*Nx+j,i*Nx+j+1]+=1/dely**2

    K[i*Nx+j,i*Nx+j]+=-2/dely**2

    if j>0:
        K[i*Nx+j,i*Nx+j-1]+=1/dely**2

for r in range(Nx):
    for j in range(Nx*Ny):
        K[r,j],K[Nx*Ny-1-r,j]=0,0
        K[j,r],K[j,Nx*Ny-1-r]=0,0


def fexpli(t,vec):
    return vec+delt*np.dot(K,vec)

def aux_euler_solve(acc,f,t0,dt):
    def aux(i):
        if i==0:
            return acc[0]
        acc[0] = acc[0]+f(t0+i*dt,acc[0])*dt
        return acc[0]
    return aux

def aux_euler_predict(x0,f,t0,dt):
    def aux(x,i):
        if i==0:
            return x0
        return x+f(t0+i*dt,x)*dt
    return aux

def solve_euler_explicit2(f, x0, dt, t0, tf):
    """renvoie le vecteur des temps et le vecteur des solutions approchees de l equa dif"""
    Ttime = np.arange(t0,tf,dt,dtype=np.float64)
    return Ttime,(list(map(aux_euler_solve({0:x0},f,t0,dt),range(len(Ttime)))))

def solve_euler_explicit(f, x0, dt, t0, tf):
    Ttime = np.arange(t0,tf,dt,dtype=np.float64)
    S=[x0]
    for t in Ttime:
        S.append(S[-1]+dt*f(t,S[-1]))
    return Ttime,S

#suppose que grad_solve(A,Y) renvoie une solution X a l equation AX=Y
def aux_euler_implicit_solve(acc,grad_solve,I,K,t0,dt):
    def aux(i):
        acc[0] = grad_solve((I+dt*K),acc[0],acc[0])
        return acc[0]
    return aux

def solve_euler_implicit(grad_solve, K, x0, dt, t0, tf):
    """renvoie le vecteur des temps et le vecteur des solutions approchees de l equa dif"""
    Ttime = np.arange(t0,tf,dt,dtype=np.float64)
    n=len(K)
    I=np.identity(n,dtype=np.float64)
    return Ttime,list(map(aux_euler_implicit_solve({0:x0},grad_solve,I,K,t0,dt),range(len(Ttime))))
def norme_carre(x):
    return (np.dot(np.transpose(x),x))

def grad_solve(A,b,x_0,eps=0.1):
    r_0 = b - np.dot(A,x_0)
    p_0 = r_0
    x=np.copy(x_0)    
    k=0
    norme_r_0=norme_carre(r_0)
    while(norme_r_0 > eps ) and k<100:
        alpha = norme_r_0/np.transpose(p_0).dot(A).dot(p_0)
        x_1 = x + alpha * p_0
        r_1 = r_0 - alpha * A.dot(p_0)
        beta = norme_carre(r_1)/norme_r_0
        p_1 =r_1 + beta * p_0
        p_0=p_1
        x=x_1
        r_0=r_1
        k=k+1
        norme_r_0 = norme_carre(r_0)
    return(x)    

ti,mat=solve_euler_explicit(fexpli,T,10*delt,0,1000*delt)
dill.dump(mat, open('solution', 'wb'))

#ti,mat= solve_euler_implicit(grad_solve,-K,T,delt,0,100*delt)
#dill.dump(mat, open('solution2', 'wb'))