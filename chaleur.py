import numpy as np

delx=0.1
dely=0.1
delt=0.1

X=np.linspace(0,1,int(1/delx +1))
Y=np.linspace(0,1,int(1/dely +1))

def f(x,y):
    return 0.5 +np.sin(2*np.pi*x)*np.sin(2*np.pi*y) -0.5*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)
T=[]
for x in X:
    for y in Y:
        T.append(f(x,y))

T=np.array(T)

MatT=np.array([T])
K=np.array([])

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


