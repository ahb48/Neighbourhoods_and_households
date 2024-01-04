import numpy as np
from sympy import linsolve, symbols
from numpy import linalg as LA
from Transition_matrix_construct import QNeighbourhood, ConfigsConstruct

def eSolve(n1,n2,beta,alpha,rho,gamma,y0,neighbourhood):
    Q = QNeighbourhood(n1=n1,n2=n2,beta=beta,alpha=alpha,rho=rho,gamma=gamma,y0=y0,neighbourhood=neighbourhood) 
    
    configs1,n_configs1 = ConfigsConstruct(n1)
    configs2,n_configs2 = ConfigsConstruct(n2)
    if neighbourhood == 0:                                   # setting the set of configurations for whichever neighbourhood we need
        configs,n_configs = configs1,n_configs1
    else:
        configs,n_configs = configs2,n_configs2
    
    ivec = [config.count('i') for config in configs]
    absorb_ind = [i for i, e in enumerate(ivec) if e == 0] # vector of indices of configurations with no infectious indivduals i.e absorbing
    ivecSolve = list(np.empty(n_configs))
    ivecSolve[:] = ivec[:]
    Q = np.delete(Q,absorb_ind,axis = 0)           # removing the rows and columns of the absorbing states for the neighbourhood 1 Q matrix
    Q = np.delete(Q,absorb_ind,axis = 1)
    ivecSolve = np.delete(ivecSolve,absorb_ind,0)  # removing the household epidemic values corresponding absorbing states, neighbourhood i
    
    # this will need altering for differing household sizes between neighbourhoods
    e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, e17, e18, e19, e20, e21, e22, e23, e24, e25, e26, e27, e28, e29, e30, \
          e31, e32, e33, e34, e35, e36, e37, e38, e39, e40 = symbols("e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, e17, \
          e18, e19, e20, e21, e22, e23, e24, e25, e26, e27, e28, e29, e30, e31, e32, e33, e34, e35, e36, e37, e38, e39, e40")   
    e_k = np.delete([e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, e17, e18, e19, e20, \
                     e21, e22, e23, e24, e25, e26, e27, e28, e29, e30, e31, e32, e33, e34, e35, e36, e37, e38, e39, e40][:n_configs],absorb_ind,0)  
    # vector of e_ks solving for
    e_k = np.transpose(e_k)                              # take transpose
      
    # apply linear solver to system for neighbourhood i
    e2_soln = linsolve(list(np.matmul(Q,e_k) + ivecSolve), list(e_k)) 
    e2_soln = list(e2_soln)[0][0]                                                # retrieve e2
    return(e2_soln)

def Rstar(n1,n2,beta1,beta2,alpha1,alpha2,rho1,rho2,gamma,y0):
    e2_soln1 = eSolve(n1=n1,n2=n2,beta=beta1,alpha=alpha1,rho=rho1,gamma=gamma,y0=y0,neighbourhood=0)
    e2_soln2 = eSolve(n1=n1,n2=n2,beta=beta2,alpha=alpha2,rho=rho2,gamma=gamma,y0=y0,neighbourhood=1)
    R_star11 = e2_soln1*alpha1*(rho1)
    R_star12 = e2_soln1*alpha1*(1-rho1)
    R_star22 = e2_soln2*alpha2*(rho2)
    R_star21 = e2_soln2*alpha2*(1-rho2)
    NGM_star = np.array([[R_star11, R_star12],[R_star21, R_star22]], dtype=float)
    R_star = np.max(LA.eig(NGM_star)[0])
    return(R_star,R_star11,R_star12,R_star21,R_star22)

def Rstar_solve(beta,n1,n2,rho1,rho2,gamma,y0,nu):
    beta1,beta2=beta,beta
    alpha = beta/nu
    alpha1,alpha2=alpha,alpha
    
    e2_soln1 = eSolve(n1=n1,n2=n2,beta=beta1,alpha=alpha1,rho=rho1,gamma=gamma,y0=y0,neighbourhood=0)
    e2_soln2 = eSolve(n1=n1,n2=n2,beta=beta2,alpha=alpha2,rho=rho2,gamma=gamma,y0=y0,neighbourhood=1)
    R_star11 = e2_soln1*alpha1*(rho1)
    R_star12 = e2_soln1*alpha1*(1-rho1)
    R_star22 = e2_soln2*alpha2*(rho2)
    R_star21 = e2_soln2*alpha2*(1-rho2)
    NGM_star = np.array([[R_star11, R_star12],[R_star21, R_star22]], dtype=float)
    R_star = np.max(LA.eig(NGM_star)[0])
    return(R_star-0.9)


def RstarAnal(beta,gamma,nu):
    alpha = beta/nu
    return((alpha/gamma)*(1+beta/(beta+gamma))-2.4)