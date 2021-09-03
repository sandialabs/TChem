Version 2.3.0 (2021-09-02)
1. KineticModelConstData and KineticSurfModelConstData have different a template parameter, DeviceType instead of SpT(execution space); this change will require to modify old code.

Previous version

   TChem::KineticModelData kmd(chemFile, thermFile);
   const auto kmcd = kmd.createConstData<TChem::exec_space>();

New version 
   using device_type = typename Tines::UseThisDevice< TChem::exec_space >::type; 
   TChem::KineticModelData kmd(chemFile, thermFile);
   const auto kmcd = kmd.createConstData<device_type>();

2. Impl interface function have two template parameters typename ValueType, typename DeviceType.
3. We added a parser for Cantera-Yaml input file.
4. We added a constant volume ignition reactor.
5. We added a python interface, pytchem, for several of the C++ functions.6. We added C and Fortran interfaces.
7. Explicit instantiation support to various sacado slfad dimensions
8. Analytical (Sacado Jacobians) for all canonical reactor models


Version 2.1.0 (2021-02-09)

We removed the depends on Kokkoskernels and OpenBlass, instead version 2.1.0 uses Tines for linear algebra operations. 

We moved the ODE and DAE solvers from version 2.0.0 to the Tines library (https://github.com/sandialabs/Tines.git). Thus version 2.1.0 depends on Tines for time integration. 
  
We added a python interface using pybind11. This interface is under construction.  

We added batch interfaces for several quantities such as source terms and Jacobians for the plug flow reactor and for the continuously stirred tank reactor. 

Version 2.0.0 (2020-10-15) 
