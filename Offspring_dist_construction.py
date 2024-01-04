import numpy as np
from sympy import linsolve, symbols
from numpy import linalg as LA
from Transition_matrix_construct import QNeighbourhood, ConfigsConstruct


def Offspring(n1,n2,beta,alpha,rho,gamma,y0,neighbourhood,mmax):
   # mmax = 50  # we truncate the offspring distribution to no more than 50 infected households
    Q = QNeighbourhood(n1=n1,n2=n2,beta=beta,alpha=alpha,rho=rho,gamma=gamma,y0=y0,neighbourhood=neighbourhood) # constructing the transition matrix
    
    configs1,n_configs1 = ConfigsConstruct(n1)
    configs2,n_configs2 = ConfigsConstruct(n2)
    if neighbourhood == 0:                                   # setting the set of configurations for whichever neighbourhood we need
        configs,n_configs = configs1,n_configs1
    else:
        configs,n_configs = configs2,n_configs2
    
    ivec = [config.count('i') for config in configs]       # counting the number of infected individuals in each possible household configuration
    absorb_ind = [i for i, e in enumerate(ivec) if e == 0] # vector of indices of configurations with no infectious individuals i.e absorbing
    # initialising an array for no. of infectious individuals in each household config, we will remove the absorbing states
    ivecSolve = np.empty(n_configs)                       
    ivecSolve[:] = ivec[:]                         # populating with no. infectious
    Q = np.delete(Q,absorb_ind,axis = 0)           # removing the rows of the absorbing states for the neighbourhood i Q matrix
    Q = np.delete(Q,absorb_ind,axis = 1)           # columns
    ivecSolve = np.delete(ivecSolve,absorb_ind,0)  # removing the household epidemic values corresponding absorbing states, neighbourhood i
    
    Qf = Q - alpha*np.diag(ivecSolve)  # see eqn 2 in Ross 2010, we multiply this matrix by the vector of y_is
   
    # intialising vector which specifies household configs with 1 infected (will become absorbed state if recover next event)
    ivecAbs = np.empty(len(ivecSolve))  
    ivecAbs[:] = [gamma*(iv==1) for iv in ivecSolve]  # vector of ones and zeros, where ones denote 1 infected individual households
    
    y_soln = np.zeros((mmax,len(ivecAbs)))     # initialising array to contain y^(m), mth deriv of y_i (derives in colums 0,...,m-1)
    g_off = np.zeros(mmax)                     # intialising array to contain offspring distribution
    
    y_soln[0,:] = np.matmul(np.linalg.inv(Qf),-ivecAbs)    # \sum_{j \in s} q(i,j)y_j(s)=s*f(i)*y_i(s), i \in C (note absorbing state y_is can be separated out)
    IC = 0
    g_off[0] = y_soln[0,IC]    # g(0)=y_{IC}(1)
    for m in range(1,mmax):   # looping over possible no. of household offspring (until truncation)
        y_soln[m,:] = np.matmul(np.linalg.inv(Qf),m*alpha*ivecSolve*y_soln[m-1,:])   # differentiating y_i iteratively, see eqn 3 Ross 2010
        g_off[m] = ((-1)**m)/np.math.factorial(m) * y_soln[m,IC]  
 
    return(g_off)  # returns offspring distribution



def Gen_func_s(s,*params):
    n1,n2,beta,alpha,rho,gamma,y0,neighbourhood,mmax = params
    
    g_off = np.zeros(mmax)
    g_off[:] = Offspring(n1,n2,beta,alpha,rho,gamma,y0,neighbourhood,mmax)
    gen_func = np.sum([s**k*g_off[k] for k in range(mmax)])
    return gen_func - s