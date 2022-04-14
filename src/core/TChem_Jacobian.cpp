/* =====================================================================================
TChem version 2.0
Copyright (2020) NTESS
https://github.com/sandialabs/TChem

Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
certain rights in this software.

This file is part of TChem. TChem is open source software: you can redistribute it
and/or modify it under the terms of BSD 2-Clause License
(https://opensource.org/licenses/BSD-2-Clause). A copy of the licese is also
provided under the main directory

Questions? Contact Cosmin Safta at <csafta@sandia.gov>, or
           Kyungjoo Kim at <kyukim@sandia.gov>, or
           Oscar Diaz-Ibarra at <odiazib@sandia.gov>

Sandia National Laboratories, Livermore, CA, USA
===================================================================================== */
#include "TChem_Jacobian.hpp"

namespace TChem {

///      _                 _     _
///     | | __ _  ___ ___ | |__ (_) __ _ _ __  ___
///  _  | |/ _` |/ __/ _ \| '_ \| |/ _` | '_ \/ __|
/// | |_| | (_| | (_| (_) | |_) | | (_| | | | \__ \
  ///  \___/ \__,_|\___\___/|_.__/|_|\__,_|_| |_|___/
///

/// this functor is not visible to users
namespace Internal {

template<typename ControlType, typename ExecSpaceType>
struct Functor
{
  using device_type      = typename Tines::UseThisDevice<ExecSpaceType>::type;

  using func_real_type_1d_view = Tines::value_type_1d_view<real_type, device_type>;
  using func_real_type_2d_view = Tines::value_type_2d_view<real_type, device_type>;
  using func_real_type_3d_view = Tines::value_type_3d_view<real_type, device_type>;
  using func_kinetic_model_const_data = KineticModelConstData<device_type>;

  const ordinal_type level, per_team_extent;
  const func_real_type_2d_view state;
  const func_real_type_3d_view jac;
  const func_kinetic_model_const_data kmcd;

  const ControlType control;

  Functor(const ordinal_type _level,
          const ordinal_type _per_team_extent,
          const func_real_type_2d_view& _state,
          const func_real_type_3d_view& _jac,
          const func_kinetic_model_const_data& _kmcd)
    : level(_level)
    , per_team_extent(_per_team_extent)
    , state(_state)
    , jac(_jac)
    , kmcd(_kmcd)
    , control()
  {}

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const
  {
    const ordinal_type i = member.league_rank();
    const func_real_type_1d_view state_at_i =
      Kokkos::subview(state, i, Kokkos::ALL());
    const func_real_type_2d_view jac_at_i =
      Kokkos::subview(jac, i, Kokkos::ALL(), Kokkos::ALL());
    Scratch<func_real_type_1d_view> work(member.team_scratch(level),
                                         per_team_extent);

    const Impl::StateVector<func_real_type_1d_view> sv_at_i(kmcd.nSpec,
                                                            state_at_i);
    TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                      "Error: input state vector is not valid");
    {
      auto w = (real_type*)work.data();
      auto ww = func_real_type_1d_view(w, per_team_extent);
      const real_type t = sv_at_i.Temperature();
      const real_type p = sv_at_i.Pressure();
      const func_real_type_1d_view Xc = sv_at_i.MassFractions();
      Impl::Jacobian::team_invoke(
        control, member, t, p, Xc, jac_at_i, ww, kmcd);
    }
  }
};
} // namespace Internal

/// control tree overriding
namespace Impl {

/// Control object can be added to quickly implement a certain use case and
/// expect a compiler tcan reduce the code size with provided compiler
/// information
///
/// For a performance purpose, we control two things
///           jacobian             rate of progress
/// A         reacScoef == -1      reacNreac avail
/// B         reacScoef != -1      nRealNuReac, nOrdReac avail
///
/// The default case compute BB
struct ControlJacobianVarAA
{}; /// fastest and production-code work scenario
struct ControlJacobianVarAB
{};
struct ControlJacobianVarBB
{}; /// default, slow, compute all according to runtime variables

template<>
KOKKOS_INLINE_FUNCTION bool
JacobianEnableRealStoichiometricCoefficients<ControlJacobianVarAA>(
  const ControlJacobianVarAA& control)
{
  return false;
}

template<>
KOKKOS_INLINE_FUNCTION bool
JacobianEnableRealStoichiometricCoefficients<ControlJacobianVarAB>(
  const ControlJacobianVarAB& control)
{
  return false;
}

template<>
KOKKOS_INLINE_FUNCTION bool
RateOfProgressDerivativeEnableRealStoichiometricCoefficients(
  const ControlJacobianVarAA& control)
{
  return false;
}

template<>
KOKKOS_INLINE_FUNCTION bool
RateOfProgressDerivativeEnableArbitraryOrderTerms(
  const ControlJacobianVarAA& control)
{
  return false;
}
} // namespace Impl

void
Jacobian::runHostBatch( /// input
  const ordinal_type nBatch,
  const real_type_2d_view_host_type& state,
  /// output
  const real_type_3d_view_host_type& jac,
  /// const data from kinetic model
  const kinetic_model_host_type& kmcd)
{
  Kokkos::Profiling::pushRegion("TChem::Jacobian::runHostBatch");
  TCHEM_CHECK_ERROR(
    kmcd.nPlogReac > 0,
    "Error: calculation of the jacobian with PLOG reactions is not supported");

  const ordinal_type level = 1;
  const ordinal_type per_team_extent = getWorkSpaceSize(kmcd);
  const ordinal_type per_team_scratch =
    Scratch<real_type_1d_view_host_type>::shmem_size(per_team_extent);

  using policy_type = Kokkos::TeamPolicy<host_exec_space>;
  policy_type policy(nBatch, Kokkos::AUTO()); // fine
  policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

  const ordinal_type caseID =
    ((kmcd.enableRealReacScoef) * 1 + (kmcd.nRealNuReac > 0) * 10 +
     (kmcd.nOrdReac > 0) * 100);
  switch (caseID) {
    case 0: { /// 0: enableIntegerReacScoef
      using functor_type =
        Internal::Functor<Impl::ControlJacobianVarAA, host_exec_space>;
      functor_type func(level, per_team_extent, state, jac, kmcd);
      Kokkos::parallel_for(
        "TChem::Jacobian::runHostBatch(ControlJacobianVarAA)", policy, func);
      break;
    }
    case 10:
    case 100:
    case 110: { /// 10,100: enableIntegerReacScoef + realNuReaction +
                /// ArbitraryOrderTerms
      using functor_type =
        Internal::Functor<Impl::ControlJacobianVarAB, host_exec_space>;
      functor_type func(level, per_team_extent, state, jac, kmcd);
      Kokkos::parallel_for(
        "TChem::Jacobian::runHostBatch(ControlJacobianVarAB)", policy, func);
      break;
    }
    default: { /// Default case that enables both of the above
      using functor_type =
        Internal::Functor<Impl::ControlJacobianVarBB, host_exec_space>;
      functor_type func(level, per_team_extent, state, jac, kmcd);
      Kokkos::parallel_for(
        "TChem::Jacobian::runHostBatch(ControlJacobianVarBB)", policy, func);
      break;
    }
  }
  Kokkos::Profiling::popRegion();
}

void
Jacobian::runDeviceBatch( /// input
  const ordinal_type nBatch,
  const real_type_2d_view_type& state,
  /// output
  const real_type_3d_view_type& jac,
  /// const data from kinetic model
  const kinetic_model_type& kmcd)
{
  Kokkos::Profiling::pushRegion("TChem::Jacobian::runDeviceBatch");
  TCHEM_CHECK_ERROR(
    kmcd.nPlogReac > 0,
    "Error: calculation of the jacobian with PLOG reactions is not supported");

  const ordinal_type level = 1;
  const ordinal_type per_team_extent = getWorkSpaceSize(kmcd);
  const ordinal_type per_team_scratch =
    Scratch<real_type_1d_view_type>::shmem_size(per_team_extent);

  using policy_type = Kokkos::TeamPolicy<exec_space>;
  policy_type policy(nBatch, Kokkos::AUTO()); // fine
  policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

  const ordinal_type caseID =
    ((kmcd.enableRealReacScoef) * 1 + (kmcd.nRealNuReac > 0) * 10 +
     (kmcd.nOrdReac > 0) * 100);
  switch (caseID) {
    case 0: { /// :0: enableIntegerReacScoef
      using functor_type =
        Internal::Functor<Impl::ControlJacobianVarAA, exec_space>;
      functor_type func(level, per_team_extent, state, jac, kmcd);
      Kokkos::parallel_for(
        "TChem::Jacobian::runDeviceBatch(ControlJacobianVarAA)", policy, func);
      break;
    }
    case 10:
    case 100:
    case 110: { /// enableRealReacScoef
      using functor_type =
        Internal::Functor<Impl::ControlJacobianVarAB, exec_space>;
      functor_type func(level, per_team_extent, state, jac, kmcd);
      Kokkos::parallel_for(
        "TChem::Jacobian::runDeviceBatch(ControlJacobianVarAB)", policy, func);
      break;
    }
    default: { /// General case that enables both of the above
      using functor_type =
        Internal::Functor<Impl::ControlJacobianVarBB, exec_space>;
      functor_type func(level, per_team_extent, state, jac, kmcd);
      Kokkos::parallel_for(
        "TChem::Jacobian::runDeviceBatch(ControlJacobianVarBB)", policy, func);
      break;
    }
  }
  Kokkos::Profiling::popRegion();
}

} // namespace TChem
