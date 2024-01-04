import random
import numpy as np


# define the household class
class household:
    # household type is defined by the attributes neighbourhood, number susceptibe, infected, removed
    # num indicates the total number of households with this type
    # alpha, beta, gamma are epidemiological parameters that are set at initialisation
    # alpha should be an np.array with length equal to the number of neighbourhoods
    def __init__(self, nhood, s, i, r, alpha, beta, gamma, num):
       
        # household configuration, never changes
        self.nhood = nhood  # neighbourhood index
        self.s = s
        self.i = i
        self.r = r
        self.m = s + i + r     # household size         
            
        # propensities for the household configuration, only alphaSI_ex and alpha change   
        self.alphaSI_in = beta*s*i/(self.m-1)  # within household transmission rate
        self.alphaSI_ex_static = alpha*s       # static part of transmission from external contacts, must be multiplied by prob_inf 
                                               # np.array with length equal to number of neighbourhoods
        self.alphaIR = gamma*i                 # recovery rate
        # transmission from external contacts, alpha.s.prob_inf, np.array of length equal to number of neighbourhoods
        self.alphaSI_ex = self.alphaSI_ex_static   
        self.alpha = 0                             # total event rate for household configuration
        
        # total number of households of this type
        self.num = num
                
    # method that sets the transmission rate from external contacts given the probability a contact is with an infected individual prob_inf
    def set_alphaSI_ex(self, prob_inf):
        # transmission from external contacts, alpha.s.prob_inf
        # prob_inf must be an np.array with length equal to number of neighbourhoods
        self.alphaSI_ex = self.alphaSI_ex_static*prob_inf  # np.arrays, so elementwise multiplication

    # method that calculates the total event rate for houesholds of this type
    def set_alpha(self):
        # the sum of all event rates multiplied by the number of households of this type
        # alphaSI_ex is an array with length equal to number of neighborhoods so need to sum separately, otherwise get an array output
        self.alpha = sum([self.alphaSI_in, sum(self.alphaSI_ex), self.alphaIR])*self.num
     
    # elsewhere we determine that an event occurs in a particular household type, but don't know which particular event that is
    def get_event(self):
        # choose an event at random with probabilities proportional to the event rates
        # within a household the consequences of internal and external infection events are the same
        # ramdom.choices returns a list, here of length 1, of which we want the first element as a string
        event = random.choices( ["infection", "recovery"], weights = [self.alphaSI_in + sum(self.alphaSI_ex), self.alphaIR])[0] 
        return event        
    
    
# function to create dictionary keys that summarise household attributes
def make_key(n, s, i, r):
    key = str(n) + "-" + str(s) + "-" + str(i) + "-" + str(r) 
    return key  


# function to update the households dictionary following an event
# this function exploits the fact that in python many data structures (but not strings, integers) are mutable and so can 
# be changed by passing them to a function - see https://nedbatchelder.com/text/names.html
def update_households_dict(households_dict, event_household_key, event, **d):
    # households_dict is the dictionary containing all household types
    # event_household_key is the key, of that dictionary, for the household type where the event occurs
    # event is a string, the name of the event that occurs
    
    # add 1 to a different household type
    # construct the key of the household type to which 1 will be added    
    nhood = households_dict[event_household_key].nhood    # neighbourhood
    if event == 'recovery':
        # a recovery event decreases i by 1, increases r by 1
        s = households_dict[event_household_key].s          
        i = households_dict[event_household_key].i - 1       
        r = households_dict[event_household_key].r + 1                
    elif event == 'infection':
        # an infection event decreases s by 1, increases i by 1
        s = households_dict[event_household_key].s - 1          
        i = households_dict[event_household_key].i + 1       
        r = households_dict[event_household_key].r     
    
    new_key = make_key(nhood, s, i, r)

   # if this household type already exists, its key will be in the dictionary, so just add 1 to the number        
    if new_key in households_dict:
        households_dict[new_key].num += 1
    # otherwise, add it to the dictionary with an initial number of 1
    else:
        households_dict[new_key] = household(nhood, s, i, r, d['alpha'][nhood,:], d['beta'][nhood], d['gamma'], 1)       
            
    # an event always removes 1 from the total number of households of the household type where the event occurs and adds 1 to a different household         # type
    # remove 1 from the household type where the event occurs
    households_dict[event_household_key].num -= 1 
    # if that household type is now empty, remove it from the dictionary
    if (households_dict[event_household_key].num == 0):
        households_dict.pop(event_household_key)     
    # if household type is completely recovered, remove from dictionary 
   # return households_dict

# function to update the total infected population following an infection or recovery event
# total_inf and prob_inf are both scalars, so immutable, which means we have to return to update
def update_total_inf(total_inf, total_rec, total_pop, prob_inf, event):
    if event == 'recovery':
        total_inf -= 1
        total_rec += 1

    elif event == 'infection':
        total_inf += 1
        
    prob_inf = total_inf/total_pop
    
    return total_inf, prob_inf, total_rec


# function that removes null trajectories from output
def remove_null_trajectories(all_runs, outbreak_thresh,runs):
    
    rm = []   # array where rows to be removed are indicated with 0
    # for each run
    for iRun in range(runs):
        # check if the maximum number of infections in at least one neighborhood exceeds the outbreak threshold
        if np.max(all_runs[iRun]) < outbreak_thresh:
        # if it doesn't, indicate that it should be removed
                rm.append(iRun)
            
            
    valid_runs = []
    for iRun in range(runs):
        if iRun not in rm:
            valid_runs.append(all_runs[iRun])
    #valid_runs = all_runs[~rm]
        
    return valid_runs


def initialise(**d):
    inf_indx = random.sample([0,1,2,3,4,5],1)
    inf0 = np.zeros(6)
    inf0[inf_indx] = 1
    print(d['alpha'])
    
    # neighbourhood 0
    key = make_key(0, d['n'][0], 0, 0) # susceptibles n1, infected 0, removed 0
    # add h1 - Inf1 households of this type 
    households_dict = {key: household(0, d['n'][0], 0, 0, d['alpha'][0,:], d['beta'][0], d['gamma'], d['h'][0]-inf0[0])}
    # make a dictionary key corresponding to infected households
    key = make_key(0, d['n'][0]-1, 1, 0)
    # add Inf1 households of this type
    households_dict[key] = household(0, d['n'][0]-1, 1, 0, d['alpha'][0,:], d['beta'][0], d['gamma'], inf0[0])

    # neighbourhood 1
    key = make_key(1, d['n'][1], 0, 0) # susceptibles n1, infected 0, removed 0
    # add h2 - Inf2 households of this type
    households_dict[key] = household(1, d['n'][1], 0, 0,  d['alpha'][1,:], d['beta'][1], d['gamma'], d['h'][1]-inf0[1]) 
    # make a dictionary key corresponding to infected households
    key = make_key(1, d['n'][1]-1, 1, 0)
    # add Inf2 households of this type
    households_dict[key] = household(1, d['n'][1]-1, 1, 0, d['alpha'][1,:], d['beta'][1], d['gamma'], inf0[1])
    
    # neighbourhood 2
    key = make_key(2, d['n'][2], 0, 0) # susceptibles n1, infected 0, removed 0
    # add h2 - Inf2 households of this type
    households_dict[key] = household(2, d['n'][2], 0, 0,  d['alpha'][2,:], d['beta'][2], d['gamma'], d['h'][2]-inf0[2]) 
    # make a dictionary key corresponding to infected households
    key = make_key(2, d['n'][2]-1, 1, 0)
    # add Inf2 households of this type
    households_dict[key] = household(2, d['n'][2]-1, 1, 0, d['alpha'][2,:], d['beta'][2], d['gamma'], inf0[2])
    
    # neighbourhood 3
    key = make_key(3, d['n'][3], 0, 0) # susceptibles n1, infected 0, removed 0
    # add h2 - Inf2 households of this type
    households_dict[key] = household(3, d['n'][3], 0, 0,  d['alpha'][3,:], d['beta'][3], d['gamma'], d['h'][3]-inf0[3]) 
    # make a dictionary key corresponding to infected households
    key = make_key(3, d['n'][3]-1, 1, 0)
    # add Inf2 households of this type
    households_dict[key] = household(3, d['n'][3]-1, 1, 0, d['alpha'][3,:], d['beta'][3], d['gamma'], inf0[3])
    
     # neighbourhood 4
    key = make_key(4, d['n'][4], 0, 0) # susceptibles n1, infected 0, removed 0
    # add h2 - Inf2 households of this type
    households_dict[key] = household(4, d['n'][4], 0, 0,  d['alpha'][4,:], d['beta'][4], d['gamma'], d['h'][4]-inf0[4]) 
    # make a dictionary key corresponding to infected households
    key = make_key(4, d['n'][4]-1, 1, 0)
    # add Inf2 households of this type
    households_dict[key] = household(4, d['n'][4]-1, 1, 0, d['alpha'][4,:], d['beta'][4], d['gamma'], inf0[4])
    
     # neighbourhood 5
    key = make_key(5, d['n'][5], 0, 0) # susceptibles n1, infected 0, removed 0
    # add h2 - Inf2 households of this type
    households_dict[key] = household(5, d['n'][5], 0, 0,  d['alpha'][5,:], d['beta'][5], d['gamma'], d['h'][5]-inf0[5]) 
    # make a dictionary key corresponding to infected households
    key = make_key(5, d['n'][5]-1, 1, 0)
    # add Inf2 households of this type
    households_dict[key] = household(5, d['n'][5]-1, 1, 0, d['alpha'][5,:], d['beta'][5], d['gamma'], inf0[5])
     
    # set the total number infected in the population, and probabilty randomly contacted individual is infected
    total_inf = np.array(inf0)                                     # total number infected
    total_rec = np.zeros(len(d['n']))
    # total population size
    total_pop = np.array([d['n'][0]*d['h'][0], d['n'][1]*d['h'][1],d['n'][2]*d['h'][2],d['n'][3]*d['h'][3],d['n'][4]*d['h'][4],d['n'][5]*d['h'][5]])    
    prob_inf = total_inf/total_pop                                      # probability random contact is infected

    t = 0
    iOut = 1
    tOut = d['time_points'][iOut]

    t_out = np.zeros(len(d['time_points']))
    inf_out = np.zeros((len(d['time_points']), len(d['n'])))
    rec_out = np.zeros((len(d['time_points']), len(d['n'])))
    hhold_infed_out = np.zeros((len(d['time_points']),len(d['n'])))
    
    t_out[0] = t
    inf_out[0,:] = total_inf
    hhold_infed_out[0] = total_inf
    
    tEnd = d['tEnd']
    time_points = d['time_points']
    
    return households_dict, total_pop, total_inf, total_rec, hhold_infed_out, prob_inf, t, tOut, iOut, t_out, inf_out, rec_out, \
                       tEnd, time_points,inf_indx


# intialise the population of households
# make a dictionary key corresponding to uninfected households
def do_Gillespie(params_dict):
    
    # initialise
    obs_out_vec = []
    obs_out = 7
    neighbourhoods = [0,1,2,3,4,5]
    inf_seq = []
    households_dict, total_pop, total_inf, total_rec, hhold_infed_out, prob_inf, t, tOut, iOut, t_out, inf_out, rec_out, \
            tEnd, time_points,inf_indx = initialise(**params_dict)
    
    #  iterate    
    while t >= 0:
    # for each household type calculate the transmission rate due to external contact and the total event rate
        for value in households_dict.values():
            value.set_alphaSI_ex(prob_inf)    # update the transmission rate due to external contact with the current infected contact probability 
            value.set_alpha()                 # update the total event rate
    
        # get the reaction propensities for each household type
        households_list = list(households_dict.items())   # make a list of the key-value pairs in the households dictionary
        # extract the keys, could use .keys and .values instead of .items() but not sure if order is preserved
        keys_list = [x[0] for x in households_list] 
        values_list = [x[1] for x in households_list]     # extract the values (objects of class household)
        alpha_list = [x.alpha for x in values_list]       # extract the propensities from the household objects
            
        # calculate the time until the next reaction    
        alpha0 = sum(alpha_list)       # total reaction rate
        u = np.random.random()         # uniform random number from (0, 1)
        tau = 1/alpha0 * np.log(1/u)   # time to next reaction

        # determine the household type involved in the next reaction
        # randomly select household type key from list, where the selection weights are given by the event propensities for each household type
        # random.choices returns a list, here of length 1, we just want the first element
        event_household_key = random.choices(keys_list, weights = alpha_list)[0]  
    
        # we now know the household type in which the event occurred, next find the specific event i.e. infection or recovery    
        event = households_dict[event_household_key].get_event()   # get_event() is a method of the household class
        # now we know the event, update the households dictionary
        update_households_dict(households_dict, event_household_key, event, **params_dict) 
        
        i = int(event_household_key[0])  # neighbourhood in which event occurs is first character of key
        # and update the total number infected
        total_inf[i], prob_inf[i], total_rec[i] = update_total_inf(total_inf[i], total_rec[i], total_pop[i], \
                                                prob_inf[i], event)
        t = t + tau # update the time
        
        
        
        # stop if there are no more infected
        if sum(total_inf) == 0 or (total_inf[0]>=40 and total_inf[1]>=40 and total_inf[2]>=40 and total_inf[3]>=40 \
                                   and total_inf[4]>=40 and total_inf[5]>=40): 
            break  
        
        ## counting the number of infected households
        households_list_out = list(households_dict.items())   # make a list of the key-value pairs in the households dictionary
        # extract the keys, could use .keys and .values instead of .items() but not sure if order is preserved
        keys_list_out = [x[0] for x in households_list_out] 
        keys_list_neigh1 = list(filter(None,[(key[0]=='0')*(key[2]!=str(params_dict['n'][0]))*key for key in keys_list_out]))
        keys_list_neigh2 = list(filter(None,[(key[0]=='1')*(key[2]!=str(params_dict['n'][1]))*key for key in keys_list_out]))
        keys_list_neigh3 = list(filter(None,[(key[0]=='2')*(key[2]!=str(params_dict['n'][2]))*key for key in keys_list_out]))
        keys_list_neigh4 = list(filter(None,[(key[0]=='3')*(key[2]!=str(params_dict['n'][3]))*key for key in keys_list_out]))
        keys_list_neigh5 = list(filter(None,[(key[0]=='4')*(key[2]!=str(params_dict['n'][4]))*key for key in keys_list_out]))
        keys_list_neigh6 = list(filter(None,[(key[0]=='5')*(key[2]!=str(params_dict['n'][5]))*key for key in keys_list_out]))
        hhold_infed_out[iOut,0] = np.sum([households_dict[key_neigh1].num for key_neigh1 in keys_list_neigh1])
        hhold_infed_out[iOut,1] = np.sum([households_dict[key_neigh2].num for key_neigh2 in keys_list_neigh2])
        hhold_infed_out[iOut,2] = np.sum([households_dict[key_neigh3].num for key_neigh3 in keys_list_neigh3])
        hhold_infed_out[iOut,3] = np.sum([households_dict[key_neigh4].num for key_neigh4 in keys_list_neigh4])
        hhold_infed_out[iOut,4] = np.sum([households_dict[key_neigh5].num for key_neigh5 in keys_list_neigh5])
        hhold_infed_out[iOut,5] = np.sum([households_dict[key_neigh6].num for key_neigh6 in keys_list_neigh6])
        
      #  if (total_inf[0]>=6 or total_inf[1]>=6 or total_inf[2]>=6 or total_inf[3]>=6 or \
      #      total_inf[4]>=6 or total_inf[5]>=6): 
      #      obs_out = np.where(total_inf[:]>=6)[0][0]
      #      break
        
      #  if (hhold_infed_out[iOut,0]>=6 or hhold_infed_out[iOut,1]>=6 or hhold_infed_out[iOut,2]>=6 or hhold_infed_out[iOut,3]>=6 or \
      #      hhold_infed_out[iOut,4]>=6 or hhold_infed_out[iOut,5]>=6): 
      #      obs_out = np.where(hhold_infed_out[iOut,:]>=6)[0][0]
      #      break
       
        if (hhold_infed_out[iOut,0]>=6 or hhold_infed_out[iOut,1]>=6 or hhold_infed_out[iOut,2]>=6 or hhold_infed_out[iOut,3]>=6 or \
            hhold_infed_out[iOut,4]>=6 or hhold_infed_out[iOut,5]>=6): 
            outbreaks = np.where(hhold_infed_out[iOut,:]>=6)[0]
            for out in range(len(outbreaks)):
                if outbreaks[out] not in obs_out_vec:
                    obs_out_vec.append(outbreaks[out])
                if len(obs_out_vec) == 6: break  
        
        #for neigh in neighbourhoods:                   # for first infection not first observed
        #    if hhold_infed_out[iOut,neigh] > 0:
        #        inf_seq.append(neigh)
        #        neighbourhoods.remove(neigh)
        #        if len(inf_seq) == 6: break
        
        # the first time t passes each output time point, save some output
        if t >= tOut:
            t_out[iOut] = t
            inf_out[iOut,:] = total_inf
            rec_out[iOut,:] = total_rec
            
            # stop if we've reached or passed the final timepoint
            if t >= tEnd: break
        
            # update the timepoint for the next output 
            iOut += 1                 
            tOut = time_points[iOut]
            
            # simulate whether infection observed
            #if event == 'infection' and random.choices([1,0],weights=[0.1,0.9])[0] == 1:
            #    obs_out = i
            #    break
       
    return obs_out_vec,inf_indx[0]                #_vec,hhold_infed_out  #_vec  #inf_seq
