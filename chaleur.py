import numpy as np
import matplotlib.pyplot as plt

delx=0.01
dely=0.01
delt=0.01
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
for i in range(1,Nx-1):
    for j in range(1,Ny-1):
            K[i*Ny +j,(i+1)*Ny+j] = 1/delx**2
            K[i*Ny +j, i*Ny +j]=-2*(1/delx**2 + 1/dely**2)
            K[i*Ny +j,(i-1)*Ny+j]=1/delx**2
            K[i*Ny +j,i*Ny +j-1]=1/dely**2

def fexpli(t,vec):
    return vec-delt*np.dot(K,vec)

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

def solve_euler_explicit(f, x0, dt, t0, tf):
    """renvoie le vecteur des temps et le vecteur des solutions approchees de l equa dif"""
    Ttime = np.arange(t0,tf,dt,dtype=np.float64)
    return Ttime,np.array(list(map(aux_euler_solve({0:x0},f,t0,dt),range(len(Ttime)))))


ti,mat=solve_euler_explicit(fexpli,T,delt,0,0.5)

plt.imshow(mat[3].reshape(Nx,Ny))

plt.show()