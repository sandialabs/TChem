#!/usr/bin/env python
# coding: utf-8

import numpy as np

data = np.genfromtxt("IgnSolution.dat", dtype=str)
Header = (data[0,:]).tolist()
solTchem = (data[1:,:]).astype(float)

n_iterations, n_variables = np.shape(solTchem)
n_samples = len(np.where(solTchem[:,0]==-1)[0])
print('Number of Samples: ',n_samples,' Number of iterations per sample: ',int(n_iterations/n_samples), ' Number of variables',n_variables  )


header_input_file = "T P"
for var in Header[6:]:
    header_input_file +=" "+var

sample=solTchem[:,6:]
Tv = np.atleast_2d(solTchem[:,Header.index('Temperature[K]')]).T
Pv = np.atleast_2d(solTchem[:,Header.index('Pressure[Pascal]')]).T

sample=np.hstack((Tv,Pv,sample))

np.savetxt('thermal_pro_sample.dat',sample,header=header_input_file,comments='')

