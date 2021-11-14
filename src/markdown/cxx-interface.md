TChem provides two types of interfaces for running simulations in parallel, ``runHostBatch`` and ``runDeviceBatch``. The ``runHostBatch`` interface uses ``Kokkos::DefaultHostExecutionSpace`` with data residing on host (CPU) memory. The ``runDeviceBatch`` interface works with ``Kokkos::DefaultExecutionSpace`` which is configured via Kokkos. Generally, the default execution space is configured with either OpenMP or CUDA. When the execution space is CUDA the data, which typically consists of the kinetic model parameters and an ensemble of state vectors, should be transferred to the device memory using ``Kokkos::deep_copy``. The example below illustrates how to compute reaction rates for several samples. It requires the kinetic model data and a collection of input state vectors. The data files are parsed in the host memory, the data is copied to the device memory. After the computations are completed, one can copy the results from the device memory to the host memory to print the output.

```
#include "TChem_Util.hpp"
#include "TChem_NetProductionRatePerMass.hpp"
#include "TChem_KineticModelData.hpp"

using ordinal_type = TChem::ordinal_type;
using real_type = TChem::real_type;
using real_type_1d_view = TChem::real_type_1d_view;
using real_type_2d_view = TChem::real_type_2d_view;

int main() {
  std::string chemFile("chem.inp");
  std::string thermFile("therm.dat");
  std::string inputFile("input.dat");
  std::string outputFile("omega.dat");

  Kokkos::initialize(argc, argv);
  {
    /// kinetic model is constructed and an object is constructed on host
    TChem::KineticModelData kmd(chemFile, thermFile);

    /// device type is created using exec_space
    using device_type = typename Tines::UseThisDevice<exec_space>::type;

    /// kinetic model data is transferred to the device memory
    const auto kmcd = TChem::createGasKineticModelConstData<device_type>(kmd);

    /// input file includes the number of samples and the size of the state vector
    ordinal_type nBatch, stateVectorSize;
    TChem::readNumberOfSamplesAndStateVectorSize(inputFile, nBatch, stateVectorSize);

    /// create a 2d array storing the state vectors
    real_type_2d_view state("StateVector", nBatch, stateVectorSize);
    auto state_host = Kokkos::create_mirror_view(state);

    /// read the input file and store them into the host array
    TChem::readStateVectors(inputFile, state_host);
    /// if execution space is host execution space, this deep copy is a soft copy
    Kokkos::deep_copy(state, state_host);

    /// output: reaction rates (omega)
    real_type_2d_view omega("ReactionRates", nBatch, kmcd.nSpec);

    /// create a parallel policy with workspace
    /// for better performance, team size must be tuned instead of using AUTO
    Kokkos::TeamPolicy<TChem::exec_space>
      policy(TChem::exec_space(), nBatch, Kokkos::AUTO());
    const ordinal_type level = 1;
    const ordinal_type per_team_extent = TChem::ReactionRates::getWorkSpaceSize(kmcd);
    const ordinal_type per_team_scratch  =
      TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
    policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

    /// computes net production rates
    TChem::NetProductionRatePerMass::runDeviceBatch
      (policy,
       state,
       omega,
       kmcd);
    TChem::exec_space().fence();

    /// optionally, one can move the omega to host memory
    auto omega_host = Kokkos::create_mirror_view(omega);
    Kokkos::deep_copy(omega_host, omega);

    /// one may want to print omega_host
    for (ordinal_type s=0;s<nBatch;++s) {
      std::cout << "Sample ID = " << s << std::endl;
      for (ordinal_type k=0;k<kmcd.nSpec;++k)
         std::cout << omega_host(s, k) << std::endl;
    }
  }
  Kokkos::finalize();

  return 0;
}
```

This workflow pattern can be applied for the other similar functions, presented in Sections: [Thermodynamic Properties](cxx-api-functhermo), [Kinetic Quantities](cxx-api-kinect_quantities),  [Homogeneous-Constant-Pressure Reactors](cxx-api-Homogeneous-Constant-Pressure-Reactors), and [Plug Flow Reactor](cxx-api-pfr_function).

The ODE and DAE systems employed by various reactor models require a different workflow. The numerical implementation of these models employs a time advance object that contains the range of time integration, time step sizes, Newton solver tolerance, etc. The following example shows the parameters that can be set for these problems.


```
#include "TChem_Util.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_IgnitionZeroD.hpp"

using ordinal_type = TChem::ordinal_type;
using real_type = TChem::real_type;
using time_advance_type = TChem::time_advance_type;

using real_type_0d_view = TChem::real_type_0d_view;
using real_type_1d_view = TChem::real_type_1d_view;
using real_type_2d_view = TChem::real_type_2d_view;

using time_advance_type_0d_view = TChem::time_advance_type_0d_view;
using time_advance_type_1d_view = TChem::time_advance_type_1d_view;

using real_type_0d_view_host = TChem::real_type_0d_view_host;
using real_type_1d_view_host = TChem::real_type_1d_view_host;
using real_type_2d_view_host = TChem::real_type_2d_view_host;

using time_advance_type_0d_view_host = TChem::time_advance_type_0d_view_host;
using time_advance_type_1d_view_host = TChem::time_advance_type_1d_view_host;

int main(int argc, char *argv[]) {
  /// input files
  std::string chemFile("chem.inp");
  std::string thermFile("therm.dat");
  std::string inputFile("input.dat");

  /// time stepping parameters
  /// the range of time begin and end
  real_type tbeg(0), tend(1);  
  /// min and max time step size
  real_type dtmin(1e-11), dtmax(1e-6);
  /// maximum number of time iterations computed in a single kernels launch
  ordinal_type num_time_iterations_per_interval(1);
  /// adaptive time stepping tolerance which is compared with the error estimator
  real_type tol_time(1e-8);
  /// new ton solver absolute and relative tolerence
  real_type atol_newton(1e-8), rtol_newton(1e-5);
  /// max number of newton iterations
  ordinal_Type max_num_newton_iterations(100);
  /// max number of time ODE kernel launch
  ordinal_type max_num_time_iterations(1e3);

  Kokkos::initialize(argc, argv);
  {
    /// kinetic model is constructed and an object is constructed on host
    TChem::KineticModelData kmd(chemFile, thermFile);

    /// device type is created using exec_space
    using device_type = typename Tines::UseThisDevice<TChem::exec_space>::type;

    /// kinetic model data is transferred to the device memory
    const auto kmcd = TChem::createGasKineticModelConstData<device_type>(kmd);


    /// input file includes the number of samples and the size of the state vector
    ordinal_type nBatch, stateVectorSize;
    TChem::readNumberOfSamplesAndStateVectorSize(inputFile, nBatch, stateVectorSize);

    /// create a 2d array storing the state vectors
    real_type_2d_view state("StateVector", nBatch, stateVectorSize);
    auto state_host = Kokkos::create_mirror_view(state);

    /// read the input file and store them into the host array
    TChem::readStateVectors(inputFile, state_host);
    /// if execution space is host execution space, this deep copy is a soft copy
    Kokkos::deep_copy(state, state_host);

    /// create time advance objects
    time_advance_type tadv_default;
    tadv_default._tbeg = tbeg;
    tadv_default._tend = tend;
    tadv_default._dt = dtmin;
    tadv_default._dtmin = dtmin;
    tadv_default._dtmax = dtmax;
    tadv_default._tol_time = tol_time;
    tadv_default._atol_newton = atol_newton;
    tadv_default._rtol_newton = rtol_newton;
    tadv_default._max_num_newton_iterations = max_num_newton_iterations;
    tadv_default._num_time_iterations_per_interval = num_time_iterations_per_interval;

    /// each sample is time-integrated independently
    time_advance_type_1d_view tadv("tadv", nBatch);
    Kokkos::deep_copy(tadv, tadv_default);

    /// for print the time evolution of species, we need a host mirror view
    auto tadv_host = Kokkos::create_mirror_view(tadv);
    auto state_host = Kokkos::create_mirror_view(state);

    /// create a parallel execution policy with workspace
    Kokkos::TeamPolicy<TChem::exec_space>
      policy(TChem::exec_space(), nBatch, Kokkos::AUTO());
    const ordinal_type level = 1;
    const ordinal_type per_team_extent = TChem::IgnitionZeroD::getWorkSpaceSize(kmcd);
    const ordinal_type per_team_scratch  =
      TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
    policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

    for (; iter < max_num_time_iterations && tsum <= tend; ++iter) {
      /// in each kernel launch, it computes the number of time iterations per
      /// interval
      TChem::IgnitionZeroD::runDeviceBatch
        (policy,
         tadv, state, /// input
         t, dt, state, /// output
         kmcd);
      Kokkos::fence();

      /// terminate this loop when all samples reach the time end
      tsum = zero;
      Kokkos::parallel_reduce(
        Kokkos::RangePolicy<TChem::exec_space>(0, nBatch),
        KOKKOS_LAMBDA(const ordinal_type &i, real_type &update) {
          tadv(i)._tbeg = t(i);
          tadv(i)._dt = dt(i);
          update += t(i);
        },
        tsum);
      Kokkos::fence();
      tsum /= nBatch;

      /// to store or print the state vectors, the data must be transferred to
      /// host memory
      Kokkos::deep_copy(tadv_host, tadv);
      Kokkos::deep_copy(state_host, state);
      UserDefinedPrintStateVector(tadv_host, state_host);
    }
  }
  Kokkos::finalize();
}
```

A similar setup can be used for the functions listed in section [Reactor Models](cxx-api-Reactor-Models).

# TChem::KineticModelData
TChem's KineticModelData has a set of functions to parse Chemkin and Cantera-YAML input files and to create objects to store the kinetic model data in the device or host memory.

## Gas-Phase Kinetic Model Setup using Chemkin Input Files
```
/// Constructor takes two input files and parse the input constructing the kinetic model with gas reactions.
///   [in] Chemkin mechfile -
///   [in] Chemkin thermofile -
KineticModelData(const std::string &mechfile,
                 const std::string &thermofile);

/// The method creates a const object that stores the kinetic model on the device memory.
///   [template] DeviceType is a pair of execution and memory spaces, abstracted
    by Kokkos::Device<ExecSpace, MemorySpace>
    ExecSpace - Kokkos execution space e.g., Serial, OpenMP, Cuda, SYCL, and HIP.
    MemorySpace - HostSpace, CudaSpace, CudaUVMSpace, SYCLDeviceUSMSpace, and HIPSpace.

template<typename DeviceType> KineticModelConstData<DeviceType> createConstData()			 
```

## Gas-Phase and Surface Kinetic Model using Chemkin Input Files

```
/// Constructor takes four input files and parse the input constructing the kinetic model with gas and surface reactions.
///   [in] Chemkin gas mechfile -
///   [in] Chemkin gas thermofile -
///   [in] Chemkin surface mechSurffile -
///   [in] Chemkin surface thermoSurffile -
KineticModelData(const std::string& mechfile,
                 const std::string& thermofile,
                 const std::string& mechSurffile,
                 const std::string& thermoSurffile);

/// The method create a const object that stores the kinetic model on the device memory.
///   [template] DeviceType is a pair of execution and memory spaces, abstracted
    by Kokkos::Device<ExecSpace, MemorySpace>
    ExecSpace - Kokkos execution space e.g., Serial, OpenMP, Cuda, SYCL, and HIP.
    MemorySpace - HostSpace, CudaSpace, CudaUVMSpace, SYCLDeviceUSMSpace, and HIPSpace.

// gas object
template<typename DeviceType> KineticModelConstData<DeviceType> createConstData()		

// surface object
template<typename DeviceType> KineticModelConstData<DeviceType> createConstSurfData()
```
## Gas Kinetic Model using Cantera-YAML Input Files
```
/// Constructor takes one input files and parse the input constructing the kinetic model with gas reactions.
///   [in] Cantera-Yaml mechfile -
///   [in] hasSurface -
KineticModelData(const std::string& mechfile,
                  const bool& hasSurface=false);

/// The method create a const object that stores the kinetic model on the device memory.
///   [template] DeviceType is a pair of execution and memory spaces, abstracted
    by Kokkos::Device<ExecSpace, MemorySpace>
    ExecSpace - Kokkos execution space e.g., Serial, OpenMP, Cuda, SYCL, and HIP.
    MemorySpace - HostSpace, CudaSpace, CudaUVMSpace, SYCLDeviceUSMSpace, and HIPSpace.

// gas object
template<typename DeviceType> KineticModelConstData<DeviceType> createConstData()		

```

## Gas-Phase and Surface Kinetic Models using Cantera-YAML Input Files

```
/// Constructor takes one input files and parse the input constructing the kinetic model with gas reactions.
///   [in] Cantera-Yaml mechfile -
///   [in] hasSurface - if mechfile contains surface phase set to true
KineticModelData(const std::string& mechfile,
                  const bool& hasSurface=false);

/// The method create a const object that stores the kinetic model on the device memory.
///   [template] DeviceType is a pair of execution and memory spaces, abstracted
    by Kokkos::Device<ExecSpace, MemorySpace>
    ExecSpace - Kokkos execution space e.g., Serial, OpenMP, Cuda, SYCL, and HIP.
    MemorySpace - HostSpace, CudaSpace, CudaUVMSpace, SYCLDeviceUSMSpace, and HIPSpace.
// gas object
template<typename DeviceType> KineticModelConstData<DeviceType> createConstData()

// surface object
template<typename DeviceType> KineticModelConstData<DeviceType> createConstSurfData()
```

# Function List

The following sections list all top-level function interfaces. Here, top-level interface functions can launch parallel kernels with a given parallel execution policy.

In these functions the device/array/kinetic-model types are defined using Tines:

```
using exec_space = Kokkos::DefaultExecutionSpace;
using host_exec_space = Kokkos::DefaultHostExecutionSpace;

// device type of host and device
using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;
using device_type      = typename Tines::UseThisDevice<exec_space>::type;

// views for device type
using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;
using real_type_2d_view_type = Tines::value_type_2d_view<real_type,device_type>;
using real_type_3d_view_type = Tines::value_type_3d_view<real_type,device_type>;

// views for host  type
using real_type_1d_view_host_type = Tines::value_type_1d_view<real_type,host_device_type>;
using real_type_2d_view_host_type = Tines::value_type_2d_view<real_type,host_device_type>;
using real_type_3d_view_host_type = Tines::value_type_3d_view<real_type,host_device_type>;

// kinetic model's object for gas phase  
using kinetic_model_type = KineticModelConstData<device_type>;
using kinetic_model_host_type = KineticModelConstData<host_device_type>;

// kinetic model's object for surface phase  
using kinetic_surf_model_type = KineticSurfModelConstData<device_type>;
using kinetic_surf_model_host_type = KineticSurfModelConstData<host_device_type>;
```
<a name="cxx-api-functhermo"></a>
## Thermodynamic Properties

This section lists all top-level function interface for thermodynamic properties. These functions are launching a parallel kernel with a given parallel execution policy.

<a name="cxx-api-SpecificHeatCapacityPerMass"></a>
### SpecificHeatCapacityPerMass

```
/// Specific heat capacity constant pressure (cp) per mass (Device)
/// ===============================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] CpMass - rank 2d array sized by nBatch x nSpec storing Cp per species
///   [out] CpMixMass - rank 1d array sized by nBatch
///   [in] kmcd -  a const object of kinetic model storing in device memory
/// #include "TChem_SpecificHeatCapacityPerMass.hpp"
SpecificHeatCapacityPerMass::runDeviceBatch(
  const team_policy_type& policy,
  const real_type_2d_view_type& state,
  const real_type_2d_view_type& CpMass,
  const real_type_1d_view_type& CpMixMass,
  const kinetic_model_type& kmcd)
```

```
/// Specific heat capacity constant pressure (cp) per mass (Host)
/// ===============================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] CpMass - rank 2d array sized by nBatch x nSpec storing Cp per species
///   [out] CpMixMass - rank 1d array sized by nBatch
///   [in] kmcd -  a const object of kinetic model storing in host memory
/// #include "TChem_SpecificHeatCapacityPerMass.hpp"
SpecificHeatCapacityPerMass::runHostBatch(
  const team_policy_type& policy,
  const real_type_2d_view_host_type& state,
  const real_type_2d_view_host_type& CpMass,
  const real_type_1d_view_host_type& CpMixMass,
  const kinetic_model_host_type& kmcd)
```

<a name="cxx-api-SpecificHeatCapacityConsVolumePerMass"></a>
### SpecificHeatCapacityConsVolumePerMass

```
/// Specific heat capacity constant volume (cv) per mass (Device)
/// ===============================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] CvMixMass - rank 1d array sized by nBatch
///   [in] kmcd -  a const object of kinetic model storing in device memory
/// #include "TChem_SpecificHeatCapacityConsVolumePerMass.hpp"
SpecificHeatCapacityConsVolumePerMass::runDeviceBatch(
  const team_policy_type& policy,
  const real_type_2d_view_type& state,
  const real_type_1d_view_type& CvMixMass,
  const kinetic_model_type& kmcd)
```

```
/// Specific heat capacity constant volume (cv) per mass (Host)
/// ===============================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] CvMixMass - rank 1d array sized by nBatch
///   [in] kmcd -  a const object of kinetic model storing in host memory
/// #include "TChem_SpecificHeatCapacityConsVolumePerMass.hpp"
SpecificHeatCapacityConsVolumePerMass::runHostBatch(
  const team_policy_type& policy,
  const real_type_2d_view_host_type& state,
  const real_type_1d_view_host_type& CvMixMass,
  const kinetic_model_host_type& kmcd)
```

<a name="cxx-api-EnthalpyMass"></a>
### EnthalpyMass
```
/// Enthalpy per mass (Device)
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] EnthalpyMass - rank 2d array sized by nBatch x nSpec storing enthalpy per species
///   [out] EnthalpyMixMass - rank 1d array sized by nBatch
///   [in] kmcd -  a const object of kinetic model storing in device memory
/// #include "TChem_EnthalpyMass.hpp"
EnthalpyMass::runDeviceBatch( /// thread block size
  typename team_policy_type& policy,
  const real_type_2d_view_type& state,
  const real_type_2d_view_type& EnthalpyMass,
  const real_type_1d_view_type& EnthalpyMixMass,
  const kinetic_model_type& kmcd);
```

```
/// Enthalpy per mass (Host)
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] EnthalpyMass - rank 2d array sized by nBatch x nSpec storing enthalpy per species
///   [out] EnthalpyMixMass - rank 1d array sized by nBatch
///   [in] kmcd -  a const object of kinetic model storing in host memory
/// #include "TChem_EnthalpyMass.hpp"
EnthalpyMass::runHostBatch( /// thread block size
  const team_policy_type& policy,
  const real_type_2d_view_host_type& state,
  const real_type_2d_view_host_type& EnthalpyMass,
  const real_type_1d_view_host_type& EnthalpyMixMass,
  const kinetic_model_host_type& kmcd);
```

<a name="cxx-api-EntropyMass"></a>
### EntropyMass
```
/// Entropy per mass (Device)
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] EntropyMass - rank 2d array sized by nBatch x nSpec storing enthalpy per species
///   [out] EntropyMixMass - rank 1d array sized by nBatch
///   [in] kmcd -  a const object of kinetic model storing in device memory
/// #include "TChem_EntropyMass.hpp"
EntropyMass::runDeviceBatch(
  const team_policy_type& policy,
  const real_type_2d_view_type& state,
  const real_type_2d_view_type& EntropyMass,
  const real_type_1d_view_type& EntropyMixMass,
  const kinetic_model_type& kmcd);
```
```
/// Entropy per mass (Host)
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] EntropyMass - rank 2d array sized by nBatch x nSpec storing enthalpy per species
///   [out] EntropyMixMass - rank 1d array sized by nBatch
///   [in] kmcd -  a const object of kinetic model storing in host memory
/// #include "TChem_EntropyMass.hpp"
EntropyMass::runHostBatch( /// thread block size
  const team_policy_type& policy,
  const real_type_2d_view_host_type& state,
  const real_type_2d_view_host_type& EntropyMass,
  const real_type_1d_view_host_type& EntropyMixMass,
  const kinetic_model_host_type& kmcd)
```

<a name="cxx-api-InternalEnergyMass"></a>
### InternalEnergyMass
```
/// Internal Energy per mass (Device)
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] InternalEnergyMass - rank 2d array sized by nBatch x nSpec storing enthalpy per species
///   [out] InternalEnergyMixMass - rank 1d array sized by nBatch
///   [in] kmcd -  a const object of kinetic model storing in device memory
/// #include "TChem_InternalEnergyMass.hpp"
InternalEnergyMass::runDeviceBatch(
  const team_policy_type& policy,
  const real_type_2d_view_type& state,
  const real_type_2d_view_type& InternalEnergyMass,
  const real_type_1d_view_type& InternalEnergyMixMass,
  const kinetic_model_type& kmcd);
```

```
/// Internal Energy per mass (Host)
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] InternalEnergyMass - rank 2d array sized by nBatch x nSpec storing enthalpy per species
///   [out] InternalEnergyMixMass - rank 1d array sized by nBatch
///   [in] kmcd -  a const object of kinetic model storing in host memory
/// #include "TChem_InternalEnergyMass.hpp"
InternalEnergyMass::runHostBatch(
  const team_policy_type& policy,
  const real_type_2d_view_host_type& state,
  const real_type_2d_view_host_type& InternalEnergyMass,
  const real_type_1d_view_host_type& InternalEnergyMixMass,
  const kinetic_model_host_type& kmcd);
```

<a name="cxx-api-kinect_quantities"></a>
## Kinetic Quantities

This section lists all top-level function interface for the computation of species production rates and rate of progresses. These functions are launching a parallel kernel with a given parallel execution policy.

### NetProductionRatesPerMass
```
/// Net Production Rates per mass (Device)
/// ==============
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] omega - rank 2d array sized by nBatch x nSpec storing reaction rates
///   [in] kmcd -  a const object of kinetic model storing in device memory
/// #include "TChem_NetProductionRatePerMass.hpp"
NetProductionRatePerMass::runDeviceBatch(
  const team_policy_type& policy,
  const real_type_2d_view_type& state,
  const real_type_2d_view_type& omega,
  const kinetic_model_type& kmcd);
```
```
/// Net Production Rates per mass (Host)
/// ==============
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] omega - rank 2d array sized by nBatch x nSpec storing reaction rates
///   [in] kmcd -  a const object of kinetic model storing in device memory
/// #include "TChem_NetProductionRatePerMass.hpp"
NetProductionRatePerMass::runHostBatch( /// input
  const real_type_2d_view_host_type& state,
  const real_type_2d_view_host_type& omega,
  const kinetic_model_host_type& kmcd)
```
###NetProductionRatesPerMole

```
/// Net Production Rates per mole (Device)
/// ==============
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] omega - rank 2d array sized by nBatch x nSpec storing reaction rates
///   [in] kmcd -  a const object of kinetic model storing in device memory
/// #include "TChem_NetProductionRatePerMole.hpp"
NetProductionRatePerMole::runDeviceBatch(
  const real_type_2d_view_type& state,
  const real_type_2d_view_type& omega,
  const kinetic_model_type& kmcd);

/// Net Production Rates per mole (Host)
/// ==============
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] omega - rank 2d array sized by nBatch x nSpec storing reaction rates
///   [in] kmcd -  a const object of kinetic model storing in host memory
/// #include "TChem_NetProductionRatePerMole.hpp"
NetProductionRatePerMole::runHostBatch(
  const real_type_2d_view_host_type& state,
  const real_type_2d_view_host_type& omega,
  const kinetic_model_host_type& kmcd);  
```
### NetProductionRateSurfacePerMass

```
/// Net Production Rates Surface per mass (Device)
/// ==============
///   [in] nBatch - number of samples
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [in] site_fraction - rank 2d array sized by nBatch x nSpec(Surface)
///   [out] omega - rank 2d array sized by nBatch x nSpec(Gas) storing reaction rates gas species
///   [out] omegaSurf - rank 2d array sized by nBatch x nSpec(Surface) storing reaction rates surface species
///   [in] kmcd -  a const object of kinetic model storing in device memory(gas phase)
///   [in] kmcdSurf -  a const object of kinetic model storing in device memory (Surface phase)
/// #include "TChem_NetProductionRateSurfacePerMass.hpp"
NetProductionRateSurfacePerMass::runDeviceBatch(
  const real_type_2d_view_type& state,
  const real_type_2d_view_type& site_fraction,
  const real_type_2d_view_type& omega,
  const real_type_2d_view_type& omegaSurf,
  const kinetic_model_type& kmcd,
  const kinetic_surf_model_type& kmcdSurf);

/// Net Production Rates Surface per mass (Host)
/// ==============
///   [in] nBatch - number of samples
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [in] site_fraction - rank 2d array sized by nBatch x nSpec(Surface)
///   [out] omega - rank 2d array sized by nBatch x nSpec(Gas) storing reaction rates gas species
///   [out] omegaSurf - rank 2d array sized by nBatch x nSpec(Surface) storing reaction rates surface species
///   [in] kmcd -  a const object of kinetic model storing in device memory(gas phase)
///   [in] kmcdSurf -  a const object of kinetic model storing in host memory (Surface phase)
/// #include "TChem_NetProductionRateSurfacePerMass.hpp"
NetProductionRateSurfacePerMass::runHostBatch(
  const real_type_2d_view_host_type& state,
  const real_type_2d_view_host_type& site_fraction,
  const real_type_2d_view_host_type& omega,
  const real_type_2d_view_host_type& omegaSurf,
  const kinetic_model_host_type& kmcd,
  const kinetic_surf_model_host_type& kmcdSurf);  
```

###NetProductionRateSurfacePerMole

```
/// Net Production Rates Surface per mole (Device)
/// ==============
///   [in] nBatch - number of samples
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [in] site_fraction - rank 2d array sized by nBatch x nSpec(Surface)
///   [out] omega - rank 2d array sized by nBatch x nSpec(Gas) storing reaction rates gas species
///   [out] omegaSurf - rank 2d array sized by nBatch x nSpec(Surface) storing reaction rates surface species
///   [in] kmcd -  a const object of kinetic model storing in device memory(gas phase)
///   [in] kmcdSurf -  a const object of kinetic model storing in device memory (Surface phase)
/// #include "TChem_NetProductionRateSurfacePerMole.hpp"
NetProductionRateSurfacePerMole::runDeviceBatch(
  const real_type_2d_view_type& state,
  const real_type_2d_view_type& site_fraction,
  const real_type_2d_view_type& omega,
  const real_type_2d_view_type& omegaSurf,
  const kinetic_model_type& kmcd,
  const kinetic_surf_model_type& kmcdSurf);

/// Net Production Rates Surface per mole (Host)
/// ==============
///   [in] nBatch - number of samples
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [in] site_fraction - rank 2d array sized by nBatch x nSpec(Surface)
///   [out] omega - rank 2d array sized by nBatch x nSpec(Gas) storing reaction rates gas species
///   [out] omegaSurf - rank 2d array sized by nBatch x nSpec(Surface) storing reaction rates surface species
///   [in] kmcd -  a const object of kinetic model storing in device memory(gas phase)
///   [in] kmcdSurf -  a const object of kinetic model storing in host memory (Surface phase)
/// #include "TChem_NetProductionRateSurfacePerMole.hpp"
NetProductionRateSurfacePerMole::runHostBatch(
  const real_type_2d_view_host_type& state,
  const real_type_2d_view_host_type& site_fraction,
  const real_type_2d_view_host_type& omega,
  const real_type_2d_view_host_type& omegaSurf,
  const kinetic_model_host_type& kmcd,
  const kinetic_surf_model_host_type& kmcdSurf);  
```

### RateOfProgress

```
/// RateOfProgress (Device)
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] RoPFor - rank2d array by nBatch x number of reaction in gas phase
///   [out] RoPFor - rank2d array by nBatch x number of reaction in gas phase
///   [in] kmcd -  a const object of kinetic model storing in device memory
/// #include "TChem_RateOfProgress.hpp"
RateOfProgress::runDeviceBatch(
  typename team_policy_type& policy,
  const real_type_2d_view_type& state,
  const real_type_2d_view_type& RoPFor,
  const real_type_2d_view_type& RoPRev,
  const kinetic_model_type& kmcd)

/// RateOfProgress (Host)
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] RoPFor - rank2d array by nBatch x number of reaction in gas phase
///   [out] RoPFor - rank2d array by nBatch x number of reaction in gas phase
///   [in] kmcd -  a const object of kinetic model storing in host memory
RateOfProgress::runHostBatch(
  const team_policy_type& policy,
  const real_type_2d_view_host_type& state,
  const real_type_2d_view_host_type& RoPFor,
  const real_type_2d_view_host_type& RoPRev,
  const kinetic_model_host_type& kmcd)  
```

<a name="cxx-api-Reactor-Models"></a>
## Reactor Models: Time Advance
This section lists all top-level function interfaces for the reactor examples.
<a name="cxx-api-IgnitionZeroD"></a>
### IgnitionZeroD
```
/// Ignition 0D (Device)
/// ===========
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] tol_newton - rank 1d array of size 2; 0 absolute tolerance and 1 relative tolerance
///   [in] tol_time - rank 2d sized by number of equations in the Ignition 0D problem x 2; column 1 w.r.t absolute tolerance and column 1 w.r.t relative tolerance  
///   [in] fac -  rank 2d array sized by nBatch x number of equations in the Ignition 0D problem; this array is use to compute numerical Jacobian  
///   [in] tadv - rank 1d array sized by nBatch storing time stepping data structure
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] t_out - rank 1d array sized by nBatch storing time when exiting the function
///   [out] dt_out - rank 1d array sized by nBatch storing time step size when exiting the function
///   [out] state_out - rank 2d array sized by nBatch x stateVectorSize storing updated state vectors
///   [in] kmcd -  a const object of kinetic model storing in device memory
#inclue "TChem_IgnitionZeroD.hpp"
IgnitionZeroD::runDeviceBatch(
  const team_policy_type& policy,
  const real_type_1d_view_type& tol_newton,
  const real_type_2d_view_type& tol_time,
  const real_type_2d_view_type& fac,
  const time_advance_type_1d_view& tadv,
  const real_type_2d_view_type& state,
  const real_type_1d_view_type& t_out,
  const real_type_1d_view_type& dt_out,
  const real_type_2d_view_type& state_out,
  const KineticModelConstData<device_type >& kmcd);

/// Ignition 0D (Host)
/// ===========
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] tol_newton - rank 1d array of size 2; 0 absolute tolerance and 1 relative tolerance
///   [in] tol_time - rank 2d sized by number of equations in the Ignition 0D problem x 2; column 1 w.r.t absolute tolerance and column 1 w.r.t relative tolerance  
///   [in] fac -  rank 2d array sized by nBatch x number of equations in the Ignition 0D problem; this array is use to compute numerical Jacobian  
///   [in] tadv - rank 1d array sized by nBatch storing time stepping data structure
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] t_out - rank 1d array sized by nBatch storing time when exiting the function
///   [out] dt_out - rank 1d array sized by nBatch storing time step size when exiting the function
///   [out] state_out - rank 2d array sized by nBatch x stateVectorSize storing updated state vectors
///   [in] kmcd -  a const object of kinetic model storing in host memory
#inclue "TChem_IgnitionZeroD.hpp"
IgnitionZeroD::runHostBatch( /// input
  const team_policy_type& policy,
  const real_type_1d_view_host_type& tol_newton,
  const real_type_2d_view_host_type& tol_time,
  const real_type_2d_view_host_type& fac,
  const time_advance_type_1d_view_host& tadv,
  const real_type_2d_view_host_type& state,
  const real_type_1d_view_host_type& t_out,
  const real_type_1d_view_host_type& dt_out,
  const real_type_2d_view_host_type& state_out,
  const KineticModelConstData<host_device_type>& kmcd);
```

<a name="cxx-api-ConstantVolumeIgnitionZeroD"></a>  
### ConstantVolumeIgnitionReactor
```
/// Constant Volume Ignition 0D (Device)
/// ===========
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] tol_newton - rank 1d array of size 2; 0 absolute tolerance and 1 relative tolerance
///   [in] tol_time - rank 2d sized by number of equations in the Constant Volume Ignition 0D problem x 2; column 1 w.r.t absolute tolerance and column 1 w.r.t relative tolerance  
///   [in] fac -  rank 2d array sized by nBatch x number of equations in the Ignition 0D problem; this array is use to compute numerical Jacobian  
///   [in] tadv - rank 1d array sized by nBatch storing time stepping data structure
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] t_out - rank 1d array sized by nBatch storing time when exiting the function
///   [out] dt_out - rank 1d array sized by nBatch storing time step size when exiting the function
///   [out] state_out - rank 2d array sized by nBatch x stateVectorSize storing updated state vectors
///   [in] kmcd -  a const object of kinetic model storing in device memory
/// #include "TChem_ConstantVolumeIgnitionReactor.hpp"
ConstantVolumeIgnitionReactor::runDeviceBatch(
  const team_policy_type& policy,
  const real_type_1d_view_type& tol_newton,
  const real_type_2d_view_type& tol_time,
  const real_type_2d_view_type& fac,
  const time_advance_type_1d_view& tadv,
  const real_type_2d_view_type& state,
  const real_type_1d_view_type& t_out,
  const real_type_1d_view_type& dt_out,
  const real_type_2d_view_type& state_out,
  const KineticModelConstData<device_type >& kmcd);

/// Constant Volume Ignition 0D (Host)
/// ===========
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] tol_newton - rank 1d array of size 2; 0 absolute tolerance and 1 relative tolerance
///   [in] tol_time - rank 2d sized by number of equations in the Constant Volume  Ignition 0D problem x 2; column 1 w.r.t absolute tolerance and column 1 w.r.t relative tolerance  
///   [in] fac -  rank 2d array sized by nBatch x number of equations in the Ignition 0D problem; this array is use in the computation of the numerical Jacobian
///   [in] tadv - rank 1d array sized by nBatch storing time stepping data structure
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] t_out - rank 1d array sized by nBatch storing time when exiting the function
///   [out] dt_out - rank 1d array sized by nBatch storing time step size when exiting the function
///   [out] state_out - rank 2d array sized by nBatch x stateVectorSize storing updated state vectors
///   [in] kmcd -  a const object of kinetic model storing in host memory
/// #include "TChem_ConstantVolumeIgnitionReactor.hpp"
ConstantVolumeIgnitionReactor::runHostBatch(
  const team_policy_type& policy,
  const real_type_1d_view_host_type& tol_newton,
  const real_type_2d_view_host_type& tol_time,
  const real_type_2d_view_host_type& fac,
  const time_advance_type_1d_view_host& tadv,
  const real_type_2d_view_host_type& state,
  const real_type_1d_view_host_type& t_out,
  const real_type_1d_view_host_type& dt_out,
  const real_type_2d_view_host_type& state_out,
  const KineticModelConstData<host_device_type>& kmcd)
```

<a name="cxx-api-PlugFlowReactor"></a>
### PlugFlowReactor
```
/// Plug Flow Reactor (Device)
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] tol_newton - rank 1d array of size 2; 0 absolute tolerance and 1 relative tolerance
///   [in] tol_time - rank 2d sized by number of equations in the Plug Flow Reactor problem x 2; column 1 w.r.t absolute tolerance and column 1 w.r.t relative tolerance
///   [in] fac -  rank 2d array sized by nBatch x number of equations in the Plug Flow Reactor problem; this array is use in the computation of the numerical Jacobian
///   [in] tadv - rank 1d array sized by nBatch storing time stepping data structure
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [in] site_fraction - rank2d array by nBatch x number of surface species
///   [in] velocity -rank1d array by nBatch
///   [out] t_out - rank 1d array sized by nBatch storing time when exiting the function
///   [out] dt_out - rank 1d array sized by nBatch storing time step size when exiting the function
///   [out] state_out - rank 2d array sized by nBatch x stateVectorSize storing updated state vectors
///   [out] site_fraction_out - rank2d array by nBatch x number of surface species
///   [out] velocity_out -rank1d array by nBatch
///   [in] kmcd -  a const object of kinetic model storing in device memory
///   [in] kmcdSurf -  a const object of surface kinetic model storing in device memory
///   [in] pfrd -  a const object that contains scenario parameter of the Plug flow reactor
#inclue "TChem_PlugFlowReactor.hpp"
PlugFlowReactor::runDeviceBatch(
  const team_policy_type& policy,
  const real_type_1d_view_type& tol_newton,
  const real_type_2d_view_type& tol_time,
  const real_type_2d_view_type& fac,
  const time_advance_type_1d_view& tadv,
  const real_type_2d_view_type& state,
  const real_type_2d_view_type& site_fraction,
  const real_type_1d_view_type& velocity,
  const real_type_1d_view_type& t_out,
  const real_type_1d_view_type& dt_out,
  const real_type_2d_view_type& state_out,
  const real_type_2d_view_type& site_fraction_out,
  const real_type_1d_view_type& velocity_out,
  const kinetic_model_type& kmcd,
  const kinetic_surf_model_type& kmcdSurf,
  const PlugFlowReactorData& pfrd);
```
<a name="cxx-api-TransientContinuousStirredTankReactor"></a>
### TransientContStirredTankReactor
```
/// Transient Continuous Stirred Tank Reactor (T-CSTR) (Device)
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] tol_newton - rank 1d array of size 2; 0 absolute tolerance and 1 relative tolerance
///   [in] tol_time - rank 2d sized by number of equations in the T-CSTR problem x 2; column 1 w.r.t absolute tolerance and column 1 w.r.t relative tolerance
///   [in] fac -  rank 2d array sized by nBatch x number of equations in the T-CSTR; this array is use in the computation of the numerical Jacobian
///   [in] tadv - rank 1d array sized by nBatch storing time stepping data structure
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [in] site_fraction - rank2d array by nBatch x number of surface species
///   [in] velocity -rank1d array by nBatch
///   [out] t_out - rank 1d array sized by nBatch storing time when exiting the function
///   [out] dt_out - rank 1d array sized by nBatch storing time step size when exiting the function
///   [out] state_out - rank 2d array sized by nBatch x stateVectorSize storing updated state vectors
///   [out] site_fraction_out - rank2d array by nBatch x number of surface species
///   [out] velocity_out -rank1d array by nBatch
///   [in] kmcd -  a const object of kinetic model storing in device memory
///   [in] kmcdSurf -  a const object of surface kinetic model storing in device memory
///   [in] cstr -  a const object that contains scenario parameter of the T-CSTR
/// #include "TChem_TransientContStirredTankReactor.hpp"
TransientContStirredTankReactor::runDeviceBatch(
  const team_policy_type& policy,
  const real_type_1d_view_type& tol_newton,
  const real_type_2d_view_type& tol_time,
  const real_type_2d_view_type& fac,
  const time_advance_type_1d_view& tadv,
  const real_type_2d_view_type& state,
  const real_type_2d_view_type& site_fraction,
  const real_type_1d_view_type& t_out,
  const real_type_1d_view_type& dt_out,
  const real_type_2d_view_type& state_out,
  const real_type_2d_view_type& site_fraction_out,
  const kinetic_model_type& kmcd,
  const kinetic_surf_model_type& kmcdSurf,
  const TransientContStirredTankReactorData<device_type>& cstr);
```

<a name="cxx-api-SimpleSurface"></a>
### SimpleSurface

```
/// Simple surface (Device)
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] tol_newton - rank 1d array of size 2; 0 absolute tolerance and 1 relative tolerance
///   [in] tol_time - rank 2d sized by number of equations in the simple surface problem x 2; column 1 w.r.t absolute tolerance and column 1 w.r.t relative tolerance
///   [in] tadv - rank 1d array sized by nBatch storing time stepping data structure
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [in] site_fraction - rank2d array by nBatch x number of surface species
///   [out] t - rank 1d array sized by nBatch storing time when exiting the function
///   [out] dt - rank 1d array sized by nBatch storing time step size when exiting the function
///   [in] fac -  rank 2d array sized by nBatch x number of equations in the Simple surface problem; this array is use in the computation of the numerical Jacobian
///   [out] site_fraction_out - rank2d array by nBatch x number of surface species
///   [in] kmcd -  a const object of kinetic model storing in device memory
///   [in] kmcdSurf -  a const object of surface kinetic model storing in device memory
/// #include "TChem_SimpleSurface.hpp"
SimpleSurface::runDeviceBatch(
  const team_policy_type& policy,
  const real_type_1d_view_type& tol_newton,
  const real_type_2d_view_type& tol_time,
  const time_advance_type_1d_view& tadv,
  const real_type_2d_view_type& state,
  const real_type_2d_view_type& site_fraction,
  const real_type_1d_view_type& t_out,
  const real_type_1d_view_type& dt_out,
  const real_type_2d_view_type& site_fraction_out,
  const real_type_2d_view_type& fac,
  const kinetic_model_type& kmcd,
  const kinetic_surf_model_type& kmcdSurf);
```


<a name="cxx-api-Homogeneous-Constant-Pressure-Reactors"></a>
## Homogeneous Constant-Pressure Reactors: RHS, Jacobians, and Smatrix
Functions to compute the RHS, Jacobians, and Smatrix for the Homogeneous Constant-Pressure Reactors in parallel.

<a name="cxx-api-SourceTerm"></a>
### SourceTerm
```
/// SourceTerm (Device)
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] SourceTerm - rank2d array by nBatch x number of species + 1 (temperature)
///   [in] kmcd -  a const object of kinetic model storing in device memory
/// #include "TChem_SourceTerm.hpp"
SourceTerm::runDeviceBatch(
  const team_policy_type& policy,
  const real_type_2d_view_type& state,
  const real_type_2d_view_type& source_term,
  const kinetic_model_type& kmcd);

/// SourceTerm (Host)
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] SourceTerm - rank2d array by nBatch x number of species + 1 (temperature)
///   [in] kmcd -  a const object of kinetic model storing in host memory
/// #include "TChem_SourceTerm.hpp"
SourceTerm::runHostBatch(
  const team_policy_type& policy,
  const real_type_2d_view_host_type& state,
  const real_type_2d_view_host_type& source_term,
  const kinetic_model_host_type& kmcd);
```

<a name="cxx-api-JacobianReduced"></a>
### JacobianReduced
```
/// JacobianReduced (Device)
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] Jacobian - rank 3d array by nBatch  x number of species + 1 x number of species + 1  
///   [in] kmcd -  a const object of kinetic model storing in device memory
/// #include "TChem_JacobianReduced.hpp"
JacobianReduced::runDeviceBatch(
  const team_policy_type& policy,
  const real_type_2d_view_type& state,
  const real_type_3d_view_type& Jacobian,
  const kinetic_model_type& kmcd);

/// JacobianReduced (Host)
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] Jacobian - rank 3d array by nBatch  x number of species + 1 x number of species + 1  
///   [in] kmcd -  a const object of kinetic model storing in host memory
/// #include "TChem_JacobianReduced.hpp"
JacobianReduced::runHostBatch(
  const team_policy_type& policy,
  const real_type_2d_view_host_type& state,
  const real_type_3d_view_host_type& Jacobian,
  const kinetic_model_host_type& kmcd)
```

<a name="cxx-api-IgnitionZeroDNumJacobian"></a>
### IgnitionZeroDNumJacobian
```
/// IgnitionZeroDNumJacobian (Device)
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] jac - rank 3d array by nBatch  x number of species + 1 x number of species + 1  
///   [out] fac - rank 2d array by nBatch  x number of species + 1
///   [in] kmcd -  a const object of kinetic model storing in device memory
/// #include "TChem_IgnitionZeroDNumJacobian.hpp"
IgnitionZeroDNumJacobian::runDeviceBatch( /// thread block size
  const team_policy_type& policy,
  const real_type_2d_view_type& state,
  const real_type_3d_view_type& jac,
  const real_type_2d_view_type& fac,
  const kinetic_model_type& kmcd);

/// IgnitionZeroDNumJacobian (Host)
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] jac - rank 3d array by nBatch  x number of species + 1 x number of species + 1  
///   [out] fac - rank 2d array by nBatch  x number of species + 1
///   [in] kmcd -  a const object of kinetic model storing in host memory
/// #include "TChem_IgnitionZeroDNumJacobian.hpp"
IgnitionZeroDNumJacobian::runHostBatch( /// thread block size
  const team_policy_type& policy,
  const real_type_2d_view_host_type& state,
  /// output
  const real_type_3d_view_host_type& jac,
  const real_type_2d_view_host_type& fac,
  /// const data from kinetic model
  const kinetic_model_host_type& kmcd)
```
<a name="cxx-api-Smatrix"></a>
###Smatrix

```
/// S Matrix (Device)
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] Smatrix - rank3d array by nBatch  x number of species + 1 x twice the number of reaction in gas phase
///   [in] kmcd -  a const object of kinetic model storing in device memory
/// #include "TChem_Smatrix.hpp"
Smatrix::runDeviceBatch( /// input
  const team_policy_type& policy,
  const real_type_2d_view_type& state,
  const real_type_3d_view_type& Smatrix,
  const kinetic_model_type& kmcd);

/// S Matrix (Host)
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] Smatrix - rank3d array by nBatch  x number of species + 1 x twice the number of reaction in gas phase
///   [in] kmcd -  a const object of kinetic model storing in host memory
/// #include "TChem_Smatrix.hpp"
Smatrix::runHostBatch( /// input
  const team_policy_type& policy,
  const real_type_2d_view_host_type& state,
  /// output
  const real_type_3d_view_host_type& Smatrix,
  /// const data from kinetic model
  const kinetic_model_host_type& kmcd)  
```


<a name="cxx-api-pfr_function"></a>
##Plug Flow Reactor
Function to compute the RHS for the Plug Flow Reactor in parallel.

<a name="cxx-api-PlugFlowReactorRHS"></a>
### PlugFlowReactorRHS
```
/// Plug Flow Reactor RHS (Device)
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [in] site_fraction - rank 2d array by nBatch x number of surface species
///   [in] velocity - rank 2d array sized by nBatch x stateVectorSize
//    [out] rhs - rank 2d array sized by nBatch x stateVectorSize
///   [in] kmcd -  a const object of kinetic model storing in device memory
///   [in] kmcdSurf -  a const object of surface kinetic model storing in device memory
///   [in] pfrd -  a const object that contains scenario parameter of the Plug flow reactor
/// #include "TChem_PlugFlowReactorRHS.hpp"
PlugFlowReactorRHS::runDeviceBatch( /// input
  const team_policy_type& policy,
  const real_type_2d_view_type& state,
  const real_type_2d_view_type& site_fraction,
  const real_type_1d_view_type& velocity,
  const real_type_2d_view_type& rhs,
  const kinetic_model_type& kmcd,
  const kinetic_surf_model_type& kmcdSurf,
  const PlugFlowReactorData& pfrd);

/// Plug Flow Reactor RHS (Host)
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [in] site_fraction - rank 2d array by nBatch x number of surface species
///   [in] velocity - rank 2d array sized by nBatch x stateVectorSize
//    [out] rhs - rank 2d array sized by nBatch x stateVectorSize
///   [in] kmcd -  a const object of kinetic model storing in host memory
///   [in] kmcdSurf -  a const object of surface kinetic model storing in host memory
///   [in] pfrd -  a const object that contains scenario parameter of the Plug flow reactor
/// #include "TChem_PlugFlowReactorRHS.hpp"
PlugFlowReactorRHS::runHostBatch( /// input
  typename UseThisTeamPolicy<host_exec_space>::type& policy,
  const real_type_2d_view_host_type& state,
  /// input
  const real_type_2d_view_host_type& site_fraction,
  const real_type_1d_view_host_type& velocity,

  /// output
  const real_type_2d_view_host_type& rhs,
  /// const data from kinetic model
  const kinetic_model_host_type& kmcd,
  /// const data from kinetic model surface
  const kinetic_surf_model_host_type& kmcdSurf,
  const PlugFlowReactorData& pfrd)
```



<a name="cxx-api-funcother"></a>
## Other Interfaces

<a name="cxx-api-InitialConditionSurface"></a>
### InitialCondSurface
```
/// InitialConditionSurface (Device)
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] tol_newton - rank 1d array of size 2; 0 absolute tolerance and 1 relative tolerance
///   [in] max_num_newton_iterations - maximum number of iteration
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [in] site_fraction - rank2d array by nBatch x number of surface species
///   [out] site_fraction_out - rank2d array by nBatch x number of surface species
///   [in] fac -  rank 2d array sized by nBatch x number of equations in the Simple surface problem; this array is use in the computation of the numerical Jacobian
///   [in] kmcd -  a const object of kinetic model storing in device memory
///   [in] kmcdSurf -  a const object of surface kinetic model storing in device memory
/// #include "TChem_InitialCondSurface.hpp"
InitialCondSurface::runDeviceBatch( /// input
  const team_policy_type& policy,
  const real_type_1d_view_type& tol_newton,
  const real_type& max_num_newton_iterations,
  const real_type_2d_view_type& state,
  const real_type_2d_view_type& site_fraction,
  const real_type_2d_view_type& site_fraction_out,
  const real_type_2d_view_type& fac,
  const kinetic_model_type& kmcd,
  const kinetic_surf_model_type& kmcdSurf);
```
