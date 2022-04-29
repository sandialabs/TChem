#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import sys
import json

savedir = "inputs" 
one_atm = 101325
Pressure= one_atm;

Temp = 1000 
YH2=0.02874974919467
YO2=0.2281708469821
YN2=1.0 - YH2 - YO2

Nparameters = 19 
ReactionNumber = np.arange(Nparameters) # change parameters for reaction 0 and 2 

Nsamples    = Nparameters*2 +1
Ndim        = Nparameters
DOE = np.ones([Nsamples, Ndim])

for ireac in range(Nparameters):
  DOE[ireac + 1 ,ireac] = 0.95
  DOE[ireac + Nparameters + 1 ,ireac] = 1.05


sample = {}
sample.update({"variable name": ["T", "P", "H2", "O2", "N2"]})
sample.update({"state vector":Nsamples * [[Temp, Pressure, YH2, YO2, YN2]]})

with open(savedir+'/sample.json', 'w') as outfile:
    json.dump(sample, outfile)

#
Nreaction = len(ReactionNumber)
A = DOE
gas = {}
gas.update({"reaction_index":ReactionNumber.tolist(),"pre_exponential_factor":A.tolist()})
with open(savedir+'/Factors.json', 'w') as outfile:
    json.dump(gas, outfile)
