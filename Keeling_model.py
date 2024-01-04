# importing required modules
import itertools
import numpy as np
#from Transition_matrix_construct import ConfigsConstruct

def ConfigsConstruct(n):
    states = ['s','i','r']
    configs = [list(x) for x in itertools.combinations_with_replacement(states, r=n)]
    n_configs = len(configs)
    return(configs, n_configs)

def Icalc(configs1,configs2,n_configs1,n_configs2,H1,H2):
    '''Calculates the proportion of individuals from neighbourhoods 1 and 2 who are infectious.'''
    I1,I2 = 0,0                               # initialising
    for k in range(n_configs1):               # summing the products the proportion of households in a given configuration
        I1 += configs1[k].count('i')*H1[k]    # and their respective no. of infecteds (neighbourhood 1)
    for k in range(n_configs2):               # neighbourhood 2
        I2 += configs2[k].count('i')*H2[k]
    return(I1, I2)

def HshiftCalc(s,i,r,H,configs,n):
    '''Finds the proportion of households in configurations with an extra susceptible and 
    one fewer infected and proportion of households with an extra infected and one fewer 
    recovered than the present configuration.'''
    H_splus_iminus = 0                                            # intialising as zero for if configuration doesn't exist
    H_iplus_rminus = 0                                            # intialising as zero for if configuration doesn't exist
    if (s<n) and (i>0):                                           # checking required configuration can exist
        index1 = configs.index(['s']*(s+1)+['i']*(i-1)+['r']*r)   # finding the index of the first required configuration
        H_splus_iminus = H[index1]                                # obtaining the required configuration using its index
    if (i<n) and (r>0):                                           # checking required configuration can exist
        index2 = configs.index(['s']*s+['i']*(i+1)+['r']*(r-1))   # finding the index of the first required configuration
        H_iplus_rminus = H[index2]                                # obtaining the required configuration using its index
    return(H_splus_iminus,H_iplus_rminus)

def StatusSizes(configs, index):
    '''Finds the number of susceptibles, infecteds and recovered in the current household 
    configuration of the current neighbourhood.'''
    s = configs[index].count('s')        # finding the no. of susceptibles in the current household configuration
    i = configs[index].count('i')        # finding the no. of infecteds in the current household configuration
    r = configs[index].count('r')        # finding the no. recovered in the current household configuration
    return(s,i,r)

def ODEHouseholds2(t, state, beta1, beta2, alpha1, alpha2, rho1, rho2, gamma, n1, n2, h1, h2):
    '''Evaluates the ODE model for households of variable sizes for each neighbourhood
     (Keeling-style formulation).'''
    # creating a list of all the possible configurations in the neighbourhoods and also finding the size of this list
    configs1,n_configs1 = ConfigsConstruct(n1)  # neighbourhood 1
    configs2,n_configs2 = ConfigsConstruct(n2)  # neighbourhood 2
                   
    H1dot = np.zeros(n_configs1)          # initialising array for H1 derivative
    H2dot = np.zeros(n_configs2)          # initialising array for H2 derivative
    
    H1 = state[0:n_configs1]                         # defining H1
    H2 = state[n_configs1:n_configs1+n_configs2]     # defining H2
    
    for l in range(n_configs1):           # looping through the household configurations in neighbourhood 1
        s,i,r = StatusSizes(configs1,l)
        H1_splus_iminus,H1_iplus_rminus = HshiftCalc(s,i,r,H1,configs1,n1)        # finding the required 'shifted' configurations
        I1,I2 = Icalc(configs1,configs2,n_configs1,n_configs2,H1,H2)           # finding the average no. of infections within a household?
        
        # ODE for neighbourhood 1, current configuration
        H1dot[l] = alpha1*((rho1)*I1/n1 + (1-rho1)*I2/n2)*(-s*H1[l] + (s+1)*H1_splus_iminus) + \
                    beta1*((-s*i/(n1-1))*H1[l] + ((s+1)*(i-1)/(n1-1))*H1_splus_iminus) + \
                    gamma*(-i*H1[l] + (i+1)*H1_iplus_rminus)
        
    for m in range(n_configs2):          # looping through the household configurations in neighbourhood 2
        s,i,r = StatusSizes(configs2,m)
        H2_splus_iminus,H2_iplus_rminus = HshiftCalc(s,i,r,H2,configs2,n2)        # finding the required 'shifted' configurations
        I1,I2 = Icalc(configs1,configs2,n_configs1,n_configs2,H1,H2)           # finding the average no. of infections within a household?
        
        # ODE for neighbourhood 2, current configuration
        H2dot[m] = alpha2*((rho2)*I2/n2 + (1-rho2)*I1/n1)*(-s*H2[m] + (s+1)*H2_splus_iminus) + \
                    beta2*((-s*i/(n2-1))*H2[m] + ((s+1)*(i-1)/(n2-1))*H2_splus_iminus) + \
                    gamma*(-i*H2[m] + (i+1)*H2_iplus_rminus)
    array_2d = np.array([H1dot,H2dot])  # putting derivatives into an array
    if n1 == n2:
        flattened = array_2d.flatten()
    else:
        flattened = np.hstack(array_2d) 
    return flattened            # flattening array to 1d