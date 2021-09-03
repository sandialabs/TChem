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
#include "TChem_MassFractionToMoleFraction.hpp"

namespace TChem {

  template<typename PolicyType,
           typename DeviceType>
void
MassFractionToMoleFraction_TemplateRun( /// input
  const std::string& profile_name,
  /// team size setting
  const PolicyType& policy,

  const Tines::value_type_2d_view<real_type, DeviceType>& state,
  /// output
  const Tines::value_type_2d_view<real_type, DeviceType>& mole_fraction,
  /// const data from kinetic model
  const KineticModelConstData<DeviceType >& kmcd)
{
  Kokkos::Profiling::pushRegion(profile_name);
  using policy_type = PolicyType;
  using device_type = DeviceType;

  using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;

  Kokkos::parallel_for(
    profile_name,
    policy,
    KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
      const ordinal_type i = member.league_rank();
      const real_type_1d_view_type state_at_i =
        Kokkos::subview(state, i, Kokkos::ALL());
      const real_type_1d_view_type Xs =
        Kokkos::subview(mole_fraction, i, Kokkos::ALL());

      const Impl::StateVector<real_type_1d_view_type> sv_at_i(kmcd.nSpec,
                                                         state_at_i);
      TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                        "Error: input state vector is not valid");
      {

        const real_type_1d_view_type Ys = sv_at_i.MassFractions();

        Impl::MassFractionToMoleFraction<real_type,device_type>::team_invoke(
          member, Ys, Xs, kmcd);
      }
    });
  Kokkos::Profiling::popRegion();
}

void
MassFractionToMoleFraction::runDeviceBatch( /// thread block size
  typename UseThisTeamPolicy<exec_space>::type& policy,
  const real_type_2d_view_type& state,
  const real_type_2d_view_type& mole_fraction,
  const kinetic_model_type& kmcd
)
{
  MassFractionToMoleFraction_TemplateRun(
  "TChem::MassFractionToMoleFraction::runDeviceBatch",
  /// team policy
  policy,
  state,
  mole_fraction,
  kmcd);
}

void
MassFractionToMoleFraction::runDeviceBatch( /// thread block size
  const ordinal_type nBatch,
  const real_type_2d_view_type& state,
  const real_type_2d_view_type& mole_fraction,
  const kinetic_model_type& kmcd)
{
  // setting policy
  const auto exec_space_instance = TChem::exec_space();
  using policy_type =
    typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;

  policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());

  MassFractionToMoleFraction_TemplateRun(
  "TChem::MassFractionToMoleFraction::runDeviceBatch",
  /// team policy
  policy,
  state,
  mole_fraction,
  kmcd);
}


void
MassFractionToMoleFraction::runHostBatch( /// thread block size
  const ordinal_type nBatch,
  const real_type_2d_view_host_type& state,
  const real_type_2d_view_host_type& mole_fraction,
  const kinetic_model_host_type& kmcd)
{
  // setting policy
  const auto exec_space_instance = TChem::host_exec_space();
  using policy_type =
    typename TChem::UseThisTeamPolicy<TChem::host_exec_space>::type;

  policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());

  MassFractionToMoleFraction_TemplateRun(
  "TChem::MassFractionToMoleFraction::runDeviceBatch",
  /// team policy
  policy,
  state,
  mole_fraction,
  kmcd);
}

void
MassFractionToMoleFraction::runHostBatch( /// thread block size
  typename UseThisTeamPolicy<host_exec_space>::type& policy,
  const real_type_2d_view_host_type& state,
  const real_type_2d_view_host_type& mole_fraction,
  const kinetic_model_host_type& kmcd
)
{
  MassFractionToMoleFraction_TemplateRun(
  "TChem::MassFractionToMoleFraction::runHostBatch",
  /// team policy
  policy,
  state,
  mole_fraction,
  kmcd);
}


} // namespace TChem
