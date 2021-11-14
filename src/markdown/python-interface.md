
TChem employs [pybind11](https://github.com/pybind/pybind11) for its Python interface to several of its C++ functions. The pytchem interface consists of a set of Python bindings for the TChem [C++ interface](#c++) "runDeviceBatch", so these computations are executed in parallel on the default execution space, i.e. OpenMP or Cuda upon its availability. In the following example, pytchem computes the net production rate (mass based) for several state vectors using the GRI3.0 mechanism. This code for this example and additional jupyter-notebooks that use pytchem are located at ``TCHEM_REPOSITORY_PATH/src/example/runs/PythonInterface``.

```python   
## set environments before importing the pytchem module.
TChem_install_directory ='/where/you/install/tchem'
import sys
sys.path.append(TChem_install_directory+'/lib')

import pytchem
import numpy as np

# Initialization of Kokkos.
pytchem.initialize()

# Create a TChem driver object.
tchem = pytchem.TChemDriver()

# Create Kinetic Model
#  Here, we use the GRIMech 3.0 gas-phase reaction mechanism
inputs_directory = TChem_install_directory + '/example/data/ignition-zero-d/gri3.0/'
tchem.createGasKineticModel(inputs_directory+'chem.inp',inputs_directory+'therm.dat')

# Set State Vector
Pressure = 101325 # Pa
Temperature = 1000 # K

# Species names must match name from kinetic model files
Variables = ['T','P','CH4','O2','N2']
Nvar = len(Variables)

# Number of samples
Nsamples = 100
# Create a temporal array to store values of variables that are non zeros
sample = np.zeros([Nsamples,Nvar])
sample[:,0] =  Temperature
sample[:,1] =  Pressure
sample[:,2] =  0.1 ## mass fraction of CH4
sample[:,3] =  0.2 ## mass fraction of O2
sample[:,4] =  0.7 ## mass fraction of N2

# Set Number of samples
tchem.setNumberOfSamples(Nsamples)

# Internally construct const object of the kinetic model and load them to device
tchem.createGasKineticModelConstData()

# Get index for variables.
indx=[]
for var in Variables:
    indx += [tchem.getStateVariableIndex(var)]

state = np.zeros([Nsamples, tchem.getLengthOfStateVector()])
for sp in range(Nsamples):
    state[sp,indx] = sample[sp,:]


# Set number of samples or nBatch
tchem.setNumberOfSamples(Nsamples)
# Allocate memory for state vector
tchem.createStateVector()
# Set state vector
tchem.setStateVector(state)

# Compute net production rate in kg/s
tchem.computeGasNetProductionRatePerMass()

# Get values for all samples
net_production_rate = tchem.getGasNetProductionRatePerMass()

print('number of samples and number of species', np.shape(net_production_rate))
print(net_production_rate)

#delete tchem object
del(tchem)
# Finalize Kokkos. This deletes the TChem object and the data stored as Kokkos views
pytchem.finalize()

```
<!-- [comment]: # (KK: consider to reduce the help output. We just want to let users to know there is help. We do not need to give all help description in this document) -->

Part of the output from help(tchem)
```python
#  Get help from TChem driver object.
help(tchem)

Help on TChemDriver in module pytchem object:

class TChemDriver(pybind11_builtins.pybind11_object)
 |  A class to manage data movement between numpy to kokkos views in TChem::Driver object
 |  
 |  Method resolution order:
 |      TChemDriver
 |      pybind11_builtins.pybind11_object
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  __init__(...)
 |      __init__(self: pytchem.TChemDriver) -> None
 |  
 |  cloneGasKineticModel(...)
 |      cloneGasKineticModel(self: pytchem.TChemDriver) -> None
 |      
 |      Internally create clones of the kinetic model
 |  
 |  computeGasEnthapyMass(...)
 |      computeGasEnthapyMass(self: pytchem.TChemDriver) -> None
 |      
 |      Compute enthalpy mass and mixture enthalpy
 |  
 ....
 ....
 ....
 |  ----------------------------------------------------------------------
 |  Static methods inherited from pybind11_builtins.pybind11_object:
 |  
 |  __new__(*args, **kwargs) from pybind11_builtins.pybind11_type
 |      Create and return a new object.  See help(type) for accurate signature.


```
