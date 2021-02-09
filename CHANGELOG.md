Version 2.1.0 (2021-02-09)

We removed the depends on Kokkoskernels and OpenBlass, instead version 2.1.0 uses Tines for linear algebra operations. 

We moved the ODE and DAE solvers from version 2.0.0 to the Tines library (https://github.com/sandialabs/Tines.git). Thus version 2.1.0 depends on Tines for time integration. 
  
We added a python interface using pybind11. This interface is under construction.  

We added batch interfaces for several quantities such as source terms and Jacobians for the plug flow reactor and for the continuously stirred tank reactor. 

Version 2.0.0 (2020-10-15) 
