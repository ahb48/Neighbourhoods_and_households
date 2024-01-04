import numpy as np
from sympy import linsolve, symbols
from numpy import linalg as LA
from Transition_matrix_construct import QNeighbourhood, ConfigsConstruct


def LST(s,n1,n2,beta,alpha,rho,gamma,y0,neighbourhood):
    s1,s2 = s
    
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

    if neighbourhood == 0:
        Qf = Q - alpha*(rho*(1-s1)+(1-rho)*(1-s2))*np.diag(ivecSolve)  # see eqn 2 in Ross 2010, we multiply this matrix by the vector of y_is
    else:
        Qf = Q - alpha*((1-rho)*(1-s1)+rho*(1-s2))*np.diag(ivecSolve)  # see eqn 2 in Ross 2010, we multiply this matrix by the vector of y_is
        
    # intialising vector which specifies household configs with 1 infected (will become absorbed state if recover next event)
    ivecAbs = np.empty(len(ivecSolve))  
    ivecAbs[:] = [gamma*(iv==1) for iv in ivecSolve]  # vector of ones and zeros, where ones denote 1 infected individual households
    
    y_soln = np.zeros(len(ivecAbs))     # initialising array to contain y^(m), mth deriv of y_i (derives in colums 0,...,m-1)
    
    y_soln[:] = np.matmul(np.linalg.inv(Qf),-ivecAbs) # \sum_{j \in s} q(i,j)y_j(s)=s*f(i)*y_i(s), i \in C (note absorbing state y_is can be separated out)
    IC = 0                  # initial condition is 1 infected, n-1 susceptible, which is the first entry
  
    return (y_soln[IC])    # returns the prob generating eqs we need to solve for the extinction probabilities


def Gen_func_s_neigh(s,*params):
    
    s1,s2 = s
    n1,n2,beta,alpha,rho,gamma,y0 = params             # unpacking parameters

    y1 = LST(s,n1,n2,beta,alpha,rho,gamma,y0,0)        # finding LST for required initial condition, y_IC, for neighbourhood 1
    y2 = LST(s,n1,n2,beta,alpha,rho,gamma,y0,1)        # finding LST for required initial condition, y_IC, for neighbourhood 2
    
    return(y1-s1,y2-s2)          # solving s1=y_IC(1-s1,1-s2), s2=y_IC(1-s1,1-s2) for extinction prob., see Ross and Black 2014.

