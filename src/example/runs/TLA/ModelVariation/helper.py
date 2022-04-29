import json
import numpy as np

def readTChemJson(file_name,tla=False):
    # read and parser json file
    f1 = open(file_name)
    data = json.load(f1)
    f1.close()
    # get variable names 
    var_names = data['variable name']
    # total number of time iterations (n_samples * number of time iterations)
    niter=len(data['state vector'])-1
    # get number of samples
    n_samples = int(data['number of samples']) 
    # ge time iteration per sample
    n = int(niter/n_samples)
    t = []
    dt = []
    sp = []
    count =0
    # loop over data
    for i in range(n):
        temp_t = []
        temp_dt = []
        temp_sp = []
        for j in range(n_samples):
            st = data['state vector'][count]
            count +=1
            temp_t += [st['t']]
            temp_dt += [st['dt']]
            temp_sp += [st['data']]
        t += [temp_t]
        dt += [temp_dt]
        sp += [temp_sp]
    # convert to np.array    
    sp = np.array(sp)
    t = np.array(t)
    dt = np.array(dt)
    if tla:
        return var_names, t, dt, sp, n_samples, data['parameter name']
    else:
        return var_names, t, dt, sp, n_samples
