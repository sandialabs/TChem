#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cbook as cbook
import sys
dirwork ='/Users/odiazib/csp_clang_bld_develop/TChem++/master/src/example/runs/scripts/'
sys.path.append(dirwork)
import pmixSample


# In[2]:
Nsamples = 2#int(sys.argv[1])
UF = 1 #float(sys.argv[2])

#sampledir = sys.argv[3]
savedir = './'#sys.argv[4]


# In[3]:
ReactionNumber = [1, 3, 5, 7, 1142, 1140, 1139]#range(Ndim) # change parameters for reaction 0 and 2 
Nparameter = len(ReactionNumber); 
Ndim=Nparameter
sampleParam = np.vstack((Ndim*[1],Ndim*[1],Ndim*[1])) # reference sample 

for sp in range(1,Nsamples):
    sampleParam = np.vstack((sampleParam, [.1,2,3,4, 1e-2, 10, 20 ], Ndim*[1], Ndim*[1]))
header =""
for Reac in ReactionNumber:
    header += str(Reac) + " "
np.savetxt(savedir + '/Factors.dat',sampleParam,header=header,comments='')

