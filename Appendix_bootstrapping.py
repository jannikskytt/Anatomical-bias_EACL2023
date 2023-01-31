# Calculate mean accuracy, confidence intervals and difference of means for 𝒯 and 𝒯⊄𝑥 

# Import necessary libraries
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
from math import sqrt

def draw_bs_replicates(data,func,size):
    """creates a bootstrap sample, computes replicates and returns replicates array"""
    # Create an empty array to store replicates
    bs_replicates = np.empty(size)
    np.random.seed(10)
    # Create bootstrap replicates as much as size
    for i in range(size):
        # Create a bootstrap sample
        bs_sample = np.random.choice(data,size=len(data))
        # Get bootstrap replicate and append to bs_replicates
        bs_replicates[i] = func(bs_sample)
    
    return bs_replicates


###########-------------- VTE ----------------####################

#Example for bootstrapping between 𝒯 and 𝒯⊄lung
𝒯 = {'Lower extrimity': [0.98, 0.98, 0.99, 0.98, 0.99], 'lung': [0.97, 0.97, 0.975, 0.98, 0.985], 'liver': [0.6904, 0.6946, 0.841, 0.7113, 0.728], 'cerebral': [0.0642, 0.0963, 0.1147, 0.0826, 0.0963], 'upper extrimity': [0.9375, 0.9602, 0.9602, 0.9489, 0.983]} 

𝒯⊄lung={'Lower extrimity': [1.0, 1.0, 0.995, 0.99, 1.0], 'lung': [0.305, 0.155, 0.11, 0.08, 0.185], 'liver': [0.8996, 0.8033, 0.6151, 0.4184, 0.5523], 'cerebral': [0.3073, 0.1468, 0.0596, 0.0505, 0.1193], 'upper extrimity': [0.9943, 0.9943, 0.9773, 0.9886, 0.9943]} 



for key in 𝒯.keys():
  data_dict = {'organ_removed_'+str(key): 𝒯⊄lung[key], 'reduced_data_'+str(key): 𝒯[key]}
  print(data_dict)

  replicates_dict = {}

  for key in data_dict:
    data = data_dict[key]
    # Draw 9999 bootstrap replicates
    bs_replicates = draw_bs_replicates(data,np.mean,9999) # 9999 is default in scipy
    replicates_dict[key] = bs_replicates

  ci_dict = {}
  conf_int=[2.5,97.5]
  #conf_int=[0.5,99.5]
  for key in replicates_dict:
    bs_replicates = replicates_dict[key]
    print(key)
    bs_mean = np.mean(bs_replicates)
    bs_se = np.std(bs_replicates)
    # Get the corresponding values of 2.5th and 97.5th percentiles
    conf_interval = np.percentile(bs_replicates,conf_int)

    print(round(bs_mean,3),'±',round(bs_se,3),'('+str(round(conf_interval[0],3))+' - '+str(round(conf_interval[1],3))+')')
    print('\n')
    ci_dict[key] = [conf_interval[0]]+[bs_mean]+[conf_interval[1]]

  significant_dict = {}
  for x in replicates_dict:
    for y in replicates_dict:
      if x==y:
        continue
      x_replicates = replicates_dict[x]
      y_replicates = replicates_dict[y]
      means_replicates = x_replicates-y_replicates

      print(x+' > '+y)

      bs_mean = np.mean(means_replicates)

      bs_se = np.std(means_replicates)

      # Get the corresponding values of 2.5th and 97.5th percentiles
      conf_interval = np.percentile(means_replicates,conf_int)

      print(round(bs_mean,3),'±',round(bs_se,3),'('+str(round(conf_interval[0],3))+' , '+str(round(conf_interval[1],3))+')')
      print('\n')
      significant_dict[x+'>'+y] = [conf_interval[0]]+[bs_mean]+[conf_interval[1]]