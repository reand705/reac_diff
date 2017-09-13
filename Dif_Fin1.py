# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# coding: utf-8
%matplotlib inline
#Imprimir Rho - Borboletas
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc ## desnec?essário
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 18})

'''
Diferenças Finitas para Equação do calor não-homogênea
Duas populações N, P (predador-presa), com resposta funcional (interação) f_n e f_p
'''

#Parametros
#D_n = 20
#D_p = 4

d_n = 0.04
d_p = 0.01
L = 10.0 #Resultados dependem disso?

#Discretização
M = 200 #espaço
N = 100000 #tempo
Tmax = 500.0

# <codecell>

# -------------------- Resposta funcional: Rosensweig-MacArthur
#Parâmetros, unidades originais (com base em Maciel, Kraenkel 2014)
'''
#Sem mud. variavel
c = 1.0
m = 0.1
a = 1.0
r = 2.0
b = 1.0
K = 5.0
def f_p(n,p):
    return c*a*p*n/(b+n) - m*p

def f_n(n,p):
    return r*n*(1 - n/K) - a*p*n/(b+n)
'''
#Com mud. variavel
c = 1.0
mu = 0.1
gamma = 2.0
kappa = 5.0
def f_p(n,p):
    return c*p*n/(1.0+n) - mu*p

def f_n(n,p):
    return gamma*n*(1 - n/kappa) - p*n/(1.0+n)
'''
# -------------------- Resposta funcional: Equação do Calor "pura"
def f_p(n,p):
    return 0

def f_n(n,p):
    return 0
'''

# <codecell>

#inicialize as populações
u = np.zeros((M,N))
v = np.zeros((M,N))
#passos
tau = 1.0*Tmax/N
h = 1.0*L/M

r_n = tau*d_n/(h*h)
r_p = tau*d_p/(h*h) #Aparentemente esses r's precisam ser menores que 1/2 para estabilidade numérica

print(r_n,r_p)
'''
#Populações iniciais - senoidal
U_0 = 1.0
V_0 = 1.0
u[:,0] = U_0*np.sin(np.pi*np.linspace(0,L,M)/L)
v[:,0] = V_0*np.sin(np.pi*np.linspace(0,L,M)/L)
'''
#Populações iniciais - step function
U_0 = 1.0
V_0 = 1.0
x = np.pi*np.linspace(0,L,M)
for i in range(0,M):
    if i >= 0.3*M and i < 0.7*M:
        u[i,0] = U_0
        v[i,0] = V_0


#Contorno (Dirichlet)
u[0,:] = np.zeros(N)
u[-1,:] = np.zeros(N)
v[0,:] = np.zeros(N)
v[-1,:] = np.zeros(N)

# <codecell>

#Discretização
for n in range(0,N-1):
    u[1:M-1,n+1] = u[1:M-1,n] + r_n*(u[0:M-2,n] - 2*u[1:M-1,n] + u[2:M+2,n]) + tau*f_n(u[1:M-1,n],v[1:M-1,n])
    v[1:M-1,n+1] = v[1:M-1,n] + r_p*(v[0:M-2,n] - 2*v[1:M-1,n] + v[2:M+2,n]) + tau*f_p(u[1:M-1,n],v[1:M-1,n])

# <codecell>

plt.plot(np.linspace(0,L,M),u[:,-541])
plt.plot(np.linspace(0,L,M),v[:,-541])

# <codecell>

t = np.linspace(0,Tmax,N)
plt.plot(t[-10000:],u[M/2 - 1,-10000:])
plt.plot(t[-10000:],v[M/2 - 1,-10000:])
#plt.axis([t[-2000],t[-1],0,10])
print(np.max(u[M/2 - 1,-10000:]))
print(np.min(u[M/2 - 1,-10000:]))

