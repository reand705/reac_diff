""" 
    This program solves numerically a predator prey system with Rosenzweig MacArthur interactions in a single patch of size L     The dynamics in the patch is given by the equations

    p_t = c * p * n / (1 + n) - m * p + dp p_xx

    n_t = g * n * (1 - n/k) - p * n /(1 + n) + dn n_xx

    p = predator population 
    n = prey population 
    
"""
from pylab import plot, ion, ylim, xlim, vlines, xlabel, ylabel, legend, title
from numpy import arange, zeros, mat, exp, ones, linalg, transpose, concatenate, sqrt, savetxt, loadtxt, pi
from scipy import sparse, linalg, integrate
import scipy.sparse.linalg

# spatial grid
def grid(num, param):
    return arange(0, param['L'] + 0.000001, num['h'])

# initial condition
def p_0(x):  # predator initial condition
    y = zeros(len(x))
    yi = len(x)/4
    yf = 3*len(x)/4
    y[yi + 1:yf] = 1.

    return y

def n_0(x):  # prey initial condition
    y = zeros(len(x))
    yi = len(x)/4
    yf = 3*len(x)/4
    y[yi + 1:yf] = 1.

    #y = 0.1*exp(-x**2/(2.))
    return y


def M1(num, param, N, p):
    ''' Builds the matrix M (equations for the predator) 

    Parameters
    ----------
    num : dict containing the discretization parameters
    param : dict containing parameters of the model
    N : number of spatial points 
    p : array containing the solution in the current time

    '''

    data_p = ones((3,N))
    
    data_p[0] = (- num['k']*param['dp']/(num['h']**2))*data_p[0]
    data_p[0,-1] = 0. 
    data_p[0,-2] = - 2.*(1 - param['ap'])/num['h'] # from boundary conditions 
    data_p[1] = data_p[1]*(1 + num['k']*param['m'] - num['k']*param['c']*p[N:2*N]/(1 + p[N:2*N])
                  + 2*num['k']*param['dp']/(num['h']**2)) 
    data_p[1,  0] = (param['ap'] + (1 - param['ap'])*3./(2.*num['h'])) # from boundary conditions    
    data_p[1, -1] = (param['ap'] + (1 - param['ap'])*3./(2.*num['h'])) # from boundary conditions
    data_p[2] = (- num['k']*param['dp']/(num['h']**2))*data_p[2]
    data_p[2,0] = 0. # from boundary conditions
    data_p[2,1] =  - 2.*(1 - param['ap'])/num['h'] # from boundary conditions

    return sparse.spdiags(data_p, [-1,0,1], 2*N, 2*N)    

def M2(num, param, N, p):
    ''' Builds the matrix M2 (equations for the prey)

    Parameters
    ----------
    num : dict containing the discretization parameters
    param : dict containing parameters of the model
    N : number of spatial points
    p : array containing the solution in the current time

    '''

    data_n = ones((3,N))
    
    data_n[0] = (- num['k']*param['dn']/(num['h']**2))*data_n[0]
    data_n[0,-1] = 0.
    data_n[0,-2] = - 2.*(1 - param['an'])/num['h'] # from boundary conditions
    data_n[1] = data_n[1]*(1 - num['k']*param['g']*(1 - p[N:2*N]/param['k']) 
                             + num['k']*p[:N]/(1 + p[N:2*N]) + 2*num['k']*param['dn']/(num['h']**2))
    data_n[1,  0] = (param['an'] + (1 - param['an'])*3./(2.*num['h'])) # from boundary conditions
    data_n[1, -1] = (param['an'] + (1 - param['an'])*3./(2.*num['h'])) # from boundary conditions
    data_n[2] = (- num['k']*param['dn']/(num['h']**2))*data_n[2]
    data_n[2,0] = 0.
    data_n[2,1] = - 2.*(1 - param['an'])/num['h'] # from boundary conditions

    return sparse.spdiags(concatenate((zeros((3,N)),data_n),1), [-1,0,1], 2*N, 2*N)    

# Matrix of the linear system of equations
def Matrix(num, N, M1, M2):
    '''Builds matrix A to be solved in the backward Euler method.

    Parameters
    ----------
    num : dict containing the problem parameters
    N : number of points in space 
    Mi : arrays containing the contributions of each dependent variable to the solution

    '''
    y = (M1 + M2)

    # boundary conditions

    # left boundary of predator 
    y[0 , 2] = (1 - param['ap'])/(2.*num['h'])

    # right boundary of predator 
    y[N - 1, N - 3] = (1 - param['ap'])/(2.*num['h'])

    # left boundary of prey
    y[N , N + 2] = (1 - param['an'])/(2.*num['h'])

    # right boundary of prey 
    y[2*N - 1, 2*N - 3] = (1 - param['an'])/(2.*num['h'])

    return y
 
# Column matrix B used to determine the next time solution of the system in the backward Euler method
def B(num, q):
    '''Builds matrix B used to solve the linear system in the backward Euler method.

    Parameters
    ----------
    num : dict containing the problem parameters
    q : array containing solution in the current time 

    '''
   
    # Boundary conditions

    # left boundary of predator 
    q[0] = 0

    # left boundary of prey  
    q[N] = 0

    # right boundary of predator 
    q[N - 1] = 0

    # right boundary of prey 
    q[2*N - 1] = 0

    return q

# Plots
def plot_predator(time, x, sol, col='r', style = '-', width = 8):
    ''' Plots the population of predator, as obtained numerically, as a function of space 

    Parameters
    ----------
    time : number of time steps 
    x : array containing the space
    sol : array containing the solution of the system for different times
    style : linestyle 
 
    '''

    plot(x, sol[time, :N], color=col, linestyle=style, linewidth = 8)

def plot_prey(time, x, sol, col='b', style = '-', width = 8):
    ''' Plots the population of prey, as obtained numerically, as a function of space 

    Parameters
    ----------
    time : number of time steps 
    x : array containing the space
    x  : array containning the space of the patch
    sol : array containing the solution of the system for different times
    style : linestyle 
 
    '''

    plot(x, sol[time, N:2*N], color=col, linestyle=style, linewidth = 8)

def plot_time_evolution_predator(x, sol, col='r',  style = '-', lab='predator'):
    ''' Plots the population of predator in the center of the patch as a function of time 

    Parameters
    ----------
    time : number of time steps 
    x  : array containning the space 
    sol : array containing the solution of the system for different times
    style : linestyle 
 
    '''

# density at the center of the patch
    plot(arange(0, num['k']*len(sol[:,int(N/2)]), num['k']), sol[:,int(N/2)], color=col, linestyle=style, label=lab)
# total population
#    x = arange(0, num['k']*len(sol[:,int(N/2)]), num['k'])
#    y = []
#    for i in range(len(x)):
#        y.append(sum(sol[i,:N]))
#    plot(x, y, color=col, linestyle=style, label=lab)

def plot_time_evolution_prey(x, sol, col='g',  style = '-', lab='prey'):
    ''' Plots the population of prey in the center of the patch as a function of time 

    Parameters
    ----------
    time : number of time steps 
    x  : array containning the space 
    sol : array containing the solution of the system for different times
    style : linestyle 
 
    '''

# density at the center of the patch
    plot(arange(0, num['k']*len(sol[:,int(N + N/2)]), num['k']), sol[:,int(N + N/2)], color=col, linestyle=style, label=lab)
# total population
#    x = arange(0, num['k']*len(sol[:,int(N + N/2)]), num['k'])
#    y = []
#    for i in range(len(x)):
#        y.append(sum(sol[i,N:])/N)
#    plot(x, y, color=col, linestyle=style, label=lab)

# Discretization parameters
num = {

       'h' : 1./50, # spatial steps
       'k' : 0.002 # time steps
}

# Parameters of the model
param = {
         'c'  : 1.,
         'm'  : .1,
         'g'  : 2.,
         'k'  : 5.,
         'dp' : .04,
         'dn' : .04,
         'L'  : 1.,
         'ap' : 1., # predator probability of leaving the patch when it is at the boundary
         'an' : .75, # prey probability of leaving the patch when it is at the boundary
}

# creating grid-points on x axis
x  = grid(num, param)

N = len(x) # numper of spatial points

# initializing de dependent variable
T = 150000  # number of time steps
v = zeros((2, 2*N))
#e = zeros((T + 1, 5*N + 6*M))

v[0,:] = concatenate((p_0(x), n_0(x)))

def solution(tempo = T, v0 = v[0,], parametros = param):
    e = zeros((int(tempo/1 + 1), 2*N))
    e[0,:] = v0
    v[0, :] = v0
    
    for i in range(1, tempo + 1):
        v[1, :] = sparse.linalg.spsolve(Matrix(num, N, 
                                        M1(num, parametros, N, v[0,:]), M2(num, parametros, N, v[0,:])), 
                                        B(num, v[0,:]))
        v[0, :] = v[1,:]
        if(not i % 1): # records the solutions every 1 iterations
            e[int(i/1),:] =  v[1, :]
            #print i
    return e




