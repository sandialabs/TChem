# C++

TChem provides two types of interface so called ``runHostBatch`` and ``runDeviceBatch`` for solving many problem instances in parallel. The ``runHostBatch`` use ``Kokkos::DefaultHostExecutionSpace`` and give data lives on host memory. On the other hand, ``runDeviceBatch`` dispatch works to ``Kokkos::DefaultExecutionSpace`` which is configured in Kokkos. In general, the default execution space is configured as OpenMP or Cuda upon its availability. When we use Cuda device, data should be transferred to the device memory using ``Kokkos::deep_copy``. An example in the below illustrates how to compute reaction rates of many samples. It reads kinetic model data and a collection of input state vectors. As the data files are available on the host memory, the input data is copied to the device memory. After computation is done, one can copy the device memory to the host memory to print the output.

```
#include "TChem_Util.hpp"
#include "TChem_ReactionRates.hpp"
#include "TChem_KineticModelData.hpp"

using ordinal_type = TChem::ordinal_type;
using real_type = TChem::real_type;
using real_type_1d_view = TChem::real_type_1d_view;
using real_type_2d_view = TChem::real_type_2d_view;

int main() {
  std::string chemFile("chem.inp");
  std::string thermFile("therm.dat");
  std::string periodictableFile("periodictable.dat");
  std::string inputFile("input.dat");
  std::string outputFile("omega.dat");

  Kokkos::initialize(argc, argv);
  {
    /// kinetic model is constructed and an object is constructed on host
    TChem::KineticModelData kmd(chemFile, thermFile, periodictableFile);

    /// kinetic model data is transferred to the device memory
    const auto kmcd = kmd.createConstData<TChem::exec_space>();

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

This pattern can be applied for the other similar functions.
* [SpecificHeatCapacityPerMass](cxx-api-SpecificHeatCapacityPerMass)
* [EnthalpyMass](cxx-api-EnthalpyMass)
* [ReactionRates](cxx-api-ReactionRates)


Time ODE and DAE systems require a different workflow from the above example. It needs a time advance object including the range of time integration, time step sizes, newton solver tolerence, etc. The following example shows parameters that a user can set for their own problems.
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
  std::string periodictableFile("periodictable.dat");
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
    TChem::KineticModelData kmd(chemFile, thermFile, periodictableFile);

    /// kinetic model data is transferred to the device memory
    const auto kmcd = kmd.createConstData<TChem::exec_space>();

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
A similar pattern can be applied for the following functions.
* [IgnitionZeroD](cxx-api-IgnitionZeroD)
* [PlugFlowReactor](cxx-api-PlugFlowReactor)

## Function List

This section lists all top-level function interface. Here, so-called top-level interface means that the function launches a parallel kernel with a given parallel execution policy.

<a name="cxx-api-SpecificHeatCapacityPerMass"></a>
### SpecificHeatCapacityPerMass
```
/// Specific heat capacity per mass
/// ===============================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] CpMass - rank 2d array sized by nBatch x nSpec storing Cp per species
///   [out] CpMixMass - rank 1d array sized by nBatch
///   [in] kmcd -  a const object of kinetic model storing in device memory
#inclue "TChem_SpecificHeatCapacityPerMass.hpp"
TChem::SpecificHeatCapacityPerMass::runDeviceBatch
  (const team_policy_type &policy,
   const real_type_2d_view &state,
   const real_type_2d_view &CpMass,
   const real_type_1d_view &CpMixMass,
   cosnt KineticModelConstDataDevice &kmcd);
```

<a name="cxx-api-SpecificHeatCapacityConsVolumePerMass"></a>
### SpecificHeatCapacityConsVolumePerMass
```
/// Specific heat capacity per mass
/// ===============================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] CvMixMass - rank 1d array sized by nBatch
///   [in] kmcd -  a const object of kinetic model storing in device memory
#inclue "TChem_SpecificHeatCapacityPerMass.hpp"
TChem::SpecificHeatCapacityPerMass::runDeviceBatch
  (const team_policy_type &policy,
   const real_type_2d_view &state,
   const real_type_1d_view &CpMixMass,
   cosnt KineticModelConstDataDevice &kmcd);
```

<a name="cxx-api-EnthalpyMass"></a>
### EnthalpyMass
```
/// Enthalpy per mass
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] EnthalpyMass - rank 2d array sized by nBatch x nSpec storing enthalpy per species
///   [out] EnthalpyMixMass - rank 1d array sized by nBatch
///   [in] kmcd -  a const object of kinetic model storing in device memory
#inclue "TChem_EnthalpyMass.hpp"
TChem::EnthalpyMass::runDeviceBatch
  (const team_policy_type &policy,
   const real_type_2d_view &state,
   const real_type_2d_view &EnthalpyMass,
   const real_type_1d_view &EnthalpyMixMass,
   cosnt KineticModelConstDataDevice &kmcd);
```

<a name="cxx-api-EntropyMass"></a>
### EntropyMass
```
/// Entropy per mass
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] EntropyMass - rank 2d array sized by nBatch x nSpec storing enthalpy per species
///   [out] EntropyMixMass - rank 1d array sized by nBatch
///   [in] kmcd -  a const object of kinetic model storing in device memory
#inclue "TChem_EntropyMass.hpp"
TChem::EntropyMass::runDeviceBatch
  (const team_policy_type &policy,
   const real_type_2d_view &state,
   const real_type_2d_view &EntropyMass,
   const real_type_1d_view &EntropyMixMass,
   cosnt KineticModelConstDataDevice &kmcd);
```

<a name="cxx-api-InternalEnergyMass"></a>
### InternalEnergyMass
```
/// Internal Energy per mass
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] InternalEnergyMass - rank 2d array sized by nBatch x nSpec storing enthalpy per species
///   [out] InternalEnergyMixMass - rank 1d array sized by nBatch
///   [in] kmcd -  a const object of kinetic model storing in device memory
#inclue "TChem_InternalEnergyMass.hpp"
TChem::InternalEnergyMass::runDeviceBatch
  (const team_policy_type &policy,
   const real_type_2d_view &state,
   const real_type_2d_view &InternalEnergyMass,
   const real_type_1d_view &InternalEnergyMixMass,
   cosnt KineticModelConstDataDevice &kmcd);
```

<a name="cxx-api-ReactionRates"></a>
### NetProductionRatesPerMass
```
/// Net Production Rates per mass
/// ==============
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] omega - rank 2d array sized by nBatch x nSpec storing reaction rates
///   [in] kmcd -  a const object of kinetic model storing in device memory
#inclue "TChem_NetProductionRatePerMass.hpp"
TChem::NetProductionRatePerMass::runDeviceBatch
  (const team_policy_type &policy,
   const real_type_2d_view &state,
   const real_type_2d_view &omega,
   const KineticModelConstDataDevice &kmcd);
```

<a name="cxx-api-ReactionRatesMole"></a>
### NetProductionRatesPerMole
```
/// Net Production Rates per mole
/// ==============
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] omega - rank 2d array sized by nBatch x nSpec storing reaction rates
///   [in] kmcd -  a const object of kinetic model storing in device memory
#inclue "TChem_NetProductionRatePerMole.hpp"
TChem::NetProductionRatePerMole::runDeviceBatch
  (const team_policy_type &policy,
   const real_type_2d_view &state,
   const real_type_2d_view &omega,
   cosnt KineticModelConstDataDevice &kmcd);
```

<a name="cxx-api-ReactionRatesSurface"></a>
### NetProductionRateSurfacePerMass
We need to update this interface in the code: OD
````
/// Net Production Rates Surface per mass
/// ==============
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [in] zSurf - rank 2d array sized by nBatch x nSpec(Surface)
///   [out] omega - rank 2d array sized by nBatch x nSpec(Gas) storing reaction rates gas species
///   [out] omegaSurf - rank 2d array sized by nBatch x nSpec(Surface) storing reaction rates surface species
///   [in] kmcd -  a const object of kinetic model storing in device memory(gas phase)
///   [in] kmcdSurf -  a const object of kinetic model storing in device memory (Surface phase)
TChem::NetProductionRateSurfacePerMass::runDeviceBatch
  (const real_type_2d_view &state,
   const real_type_2d_view &zSurf,
   const real_type_2d_view &omega,
   const real_type_2d_view &omegaSurf,
   const KineticModelConstDataDevice &kmcd,
   const KineticSurfModelConstDataDevice &kmcdSurf);

````

<a name="cxx-api-ReactionRatesSurfaceMole"></a>
### NetProductionRateSurfacePerMole
We need to update this interface in the code: OD
````
/// Net Production Rates Surface per mole
/// ==============
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [in] zSurf - rank 2d array sized by nBatch x nSpec(Surface)
///   [out] omega - rank 2d array sized by nBatch x nSpec(Gas) storing reaction rates gas species
///   [out] omegaSurf - rank 2d array sized by nBatch x nSpec(Surface) storing reaction rates surface species
///   [in] kmcd -  a const object of kinetic model storing in device memory(gas phase)
///   [in] kmcdSurf -  a const object of kinetic model storing in device memory (Surface phase)
TChem::NetProductionRateSurfacePerMole::runDeviceBatch
  (const real_type_2d_view &state,
   const real_type_2d_view &zSurf,
   const real_type_2d_view &omega,
   const real_type_2d_view &omegaSurf,
   const KineticModelConstDataDevice &kmcd,
   const KineticSurfModelConstDataDevice &kmcdSurf);

````

<a name="cxx-api-IgnitionZeroD"></a>
### Ignition 0D
```
/// Ignition 0D
/// ===========
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] tadv - rank 1d array sized by nBatch storing time stepping data structure
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] t_out - rank 1d array sized by nBatch storing time when exiting the function
///   [out] dt_out - rank 1d array sized by nBatch storing time step size when exiting the function
///   [out] state_out - rank 2d array sized by nBatch x stateVectorSize storing updated state vectors
///   [in] kmcd -  a const object of kinetic model storing in device memory
#inclue "TChem_IgnitionZeroD.hpp"
TChem::IgnitionZeroD::runDeviceBatch
  (const team_policy_type &policy,
   const time_advance_type_1d_view &tadv,
   const real_type_2d_view &state,
   const real_type_1d_view &t_out,
   const real_type_1d_view &dt_out,
   const real_type_2d_view &state_out,
   cosnt KineticModelConstDataDevice &kmcd);
```
<a name="cxx-api-PlugFlowReactor"></a>
### PlugFlowReactor
```
/// Plug Flow Reactor
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] tadv - rank 1d array sized by nBatch storing time stepping data structure
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [in] zSurf - rank2d array by nBatch x number of surface specues
///   [in] velociy -rank1d array by nBatch
///   [out] t_out - rank 1d array sized by nBatch storing time when exiting the function
///   [out] dt_out - rank 1d array sized by nBatch storing time step size when exiting the function
///   [out] state_out - rank 2d array sized by nBatch x stateVectorSize storing updated state vectors
///   [out] z_out - rank2d array by nBatch x number of surface specues
///   [out] velocity_out -rank1d array by nBatch
///   [in] kmcd -  a const object of kinetic model storing in device memory
///   [in] kmcdSurf -  a const object of surface kinetic model storing in device memory
///   [in] area - cross-sectional area
///   [in] pcat - chemically active perimeter
#inclue "TChem_PlugFlowReactor.hpp"
TChem::PlugFlowReactor::runDeviceBatch
  (const team_policy_type &policy,
   const time_advance_type_1d_view &tadv,
   const real_type_2d_view &state,
   const real_type_2d_view &z_surf,
   const real_type_1d_view &velocity,
   const real_type_1d_view &t_out,
   const real_type_1d_view &dt_out,
   const real_type_2d_view &state_out,
   const real_type_2d_view &z_out,
   const real_type_1d_view &velocity_out,
   const KineticModelConstDataDevice &kmcd,
   const KineticSurfModelConstDataDevice &kmcdSurf,
   const real_type area,
   const real_type pcat);
```

<a name="cxx-api-SimpleSurface"></a>
### SimpleSurface

```
/// Simple surface
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] tadv - rank 1d array sized by nBatch storing time stepping data structure
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [in] siteFraction - rank2d array by nBatch x number of surface species
///   [out] t - rank 1d array sized by nBatch storing time when exiting the function
///   [out] dt - rank 1d array sized by nBatch storing time step size when exiting the function
///   [out] siteFraction_out - rank2d array by nBatch x number of surface species
///   [in] kmcd -  a const object of kinetic model storing in device memory
///   [in] kmcdSurf -  a const object of surface kinetic model storing in device memory
#include "TChem_SimpleSurface.hpp"
TChem::SimpleSurface::runDeviceBatch
(const team_policy_type &policy,
 const time_advance_type_1d_view &tadv,
 const real_type_2d_view &state,
 const real_type_2d_view &siteFraction,
 const real_type_1d_view &t,
 const real_type_1d_view &dt,
 const real_type_2d_view  &siteFraction_out,
 const KineticModelConstDataDevice &kmcd,
 const KineticSurfModelConstDataDevice &kmcdSurf);

```

<a name="cxx-api-InitialConditionSurface"></a>
### InitialConditionSurface
```
/// InitialConditionSurface
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [in] siteFraction - rank2d array by nBatch x number of surface species
///   [out] siteFraction_out - rank2d array by nBatch x number of surface species
///   [in] kmcd -  a const object of kinetic model storing in device memory
///   [in] kmcdSurf -  a const object of surface kinetic model storing in device memory
#include "TChem_InitialCondSurface.hpp"
TChem::InitialCondSurface::runDeviceBatch
(const team_policy_type &policy,
 const real_type_2d_view &state,
 const real_type_2d_view &siteFraction,
 const real_type_2d_view  &siteFraction_out,
 const KineticModelConstDataDevice &kmcd,
const KineticSurfModelConstDataDevice &kmcdSurf);
```

<a name="cxx-api-RateOfProgress"></a>
### RateOfProgress
```
/// RateOfProgress
/// =================
///   [in] nBatch - number of samples
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] RoPFor - rank2d array by nBatch x number of reaction in gas phase
///   [out] RoPFor - rank2d array by nBatch x number of reaction in gas phase
///   [in] kmcd -  a const object of kinetic model storing in device memory
#include "TChem_RateOfProgress.hpp"
TChem::RateOfProgress::runDeviceBatch
( const ordinal_type nBatch,
  const real_type_2d_view& state,
  const real_type_2d_view& RoPFor,
  const real_type_2d_view& RoPRev,
  const KineticModelConstDataDevice& kmcd);
```

<a name="cxx-api-SourceTerm"></a>
### SourceTerm
```
/// SourceTerm
/// =================
///   [in] nBatch - number of samples
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] SourceTerm - rank2d array by nBatch x number of species + 1 (temperature)
///   [in] kmcd -  a const object of kinetic model storing in device memory
#include "TChem_SourceTerm.hpp"
  TChem::SourceTerm::runDeviceBatch
  (const ordinal_type nBatch,
  const real_type_2d_view& state,
  const real_type_2d_view& SourceTerm,
  const KineticModelConstDataDevice& kmcd);
```
<a name="cxx-api-Smatrix"></a>
### Smatrix
```
/// S Matrix
/// =================
///   [in] nBatch - number of samples
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] Smatrix - rank3d array by nBatch  x number of species + 1 x twice the number of reaction in gas phase
///   [in] kmcd -  a const object of kinetic model storing in device memory
#include "TChem_Smatrix.hpp"
TChem::Smatrix::runDeviceBatch
  (const ordinal_type nBatch,
   const real_type_2d_view& state,
   const real_type_3d_view& Smatrix,
   const KineticModelConstDataDevice& kmcd);
```

<a name="cxx-api-IgnitionZeroDNumJacobian"></a>
### IgnitionZeroDNumJacobian
```
/// IgnitionZeroDNumJacobian
/// =================
///   [in] nBatch - number of samples
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] jac - rank 3d array by nBatch  x number of species + 1 x number of species + 1  
///   [out] fac - rank 2d array by nBatch  x number of species + 1
///   [in] kmcd -  a const object of kinetic model storing in device memory
#include "TChem_IgnitionZeroDNumJacobian.hpp"
TChem::IgnitionZeroDNumJacobian::runDeviceBatch
  (const ordinal_type nBatch,
  const real_type_2d_view& state,
  const real_type_3d_view& jac,
  const real_type_2d_view& fac,
  const KineticModelConstDataDevice& kmcd);
```       

<a name="cxx-api-JacobianReduced"></a>
### JacobianReduced
```
/// JacobianReduced
/// =================
///   [in] nBatch - number of samples
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] Jacobian - rank 3d array by nBatch  x number of species + 1 x number of species + 1  
///   [in] kmcd -  a const object of kinetic model storing in device memory
#include "TChem_JacobianReduced.hpp"
TChem::JacobianReduced::runDeviceBatch
(const ordinal_type nBatch,
const real_type_2d_view& state,
const real_type_3d_view& Jacobian,
const KineticModelConstDataDevice& kmcd);
```
<a name="cxx-api-PlugFlowReactorRHS"></a>
### PlugFlowReactorRHS
```
/// Plug Flow Reactor RHS
/// =================
///   [in] nBatch - number of samples
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [in] zSurf - rank 2d array by nBatch x number of surface species
///   [in] velocity - rank 2d array sized by nBatch x stateVectorSize
///   [in] kmcd -  a const object of kinetic model storing in device memory
///   [in] kmcdSurf -  a const object of surface kinetic model storing in device memory

#include "TChem_PlugFlowReactorRHS.hpp"
TChem::PlugFlowReactorRHS::runDeviceBatch
(const ordinal_type nBatch,
 const real_type_2d_view& state,
 const real_type_2d_view& zSurf,
 const real_type_2d_view& velocity,
 const real_type_2d_view& rhs,
 const KineticModelConstDataDevice& kmcd,
 const KineticSurfModelConstDataDevice& kmcdSurf);
```    
