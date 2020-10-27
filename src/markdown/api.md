# Application Programming Interface

!INCLUDE "cxx-interface.md", 2

 * TChem::KineticModelData

```
/// Constructor takes three input files and parse the input constructing the kinetic model.
///   [in] mechfile -
///   [in] thermofile -
///   [in] periodictablefile 
KineticModelData(const std::string &mechfile,
                 const std::string &thermofile,
                 const std::string &periodictablefile);

/// The method create a const object that stores the kinetic model on the device memory.
///   [template] SpT - Kokkos execution space e.g., Serial, OpenMP and Cuda.
template<typename SpT> KineticModelConstData<SpT> createConstData()		 
```

 * TChem::IgnitionZeroD

```
/// The method performs time integration of multiple samples described in state.
///   [in] nBatch - # of samples
///   [in] tadv - time stepping data structure, each sample has its own time stepper
///   [in] state - a collection of state vectors corresponding to the # of samples
///   [out] t_out - time recorded when the time reaches t_adv.t_max or the max # of time integrations
///   [out[ dt_out - time step size when exit the function
///   [out] state_out - lastly updated state vetor when exit the function
///   [in] kmcd - a const object of kinetic model storing in device memory
static void runDeviceBatch(/// input                                                                      
                           const ordinal_type nBatch,
                           const time_advance_type_1d_view &tadv,
                           const real_type_2d_view &state,
                           /// output                                                                     
                           const real_type_1d_view &t_out,
                           const real_type_1d_view &dt_out,
                           const real_type_2d_view &state_out,
                           /// const data from kinetic model                                              
                           const KineticModelConstDataDevice &kmcd);
```