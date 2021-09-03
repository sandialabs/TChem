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
#include "TChem_SourceTermToyProblem.hpp"

namespace TChem {

  template<typename PolicyType,
           typename DeviceType>

  //
  void
  SourceTermToyProblem_TemplateRun( /// input
    const std::string& profile_name,
    /// team size setting
    const PolicyType& policy,
    const Tines::value_type_1d_view<real_type, DeviceType>& theta,
    const Tines::value_type_1d_view<real_type, DeviceType>& lambda,
    const Tines::value_type_2d_view<real_type, DeviceType>& state,
    const Tines::value_type_2d_view<real_type, DeviceType>& SourceTermToyProblem,
    const KineticModelConstData<DeviceType>& kmcd
  )
  {
    Kokkos::Profiling::pushRegion(profile_name);
    using policy_type = PolicyType;
    using device_type = DeviceType;

    using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;

    const ordinal_type level = 1;
    const ordinal_type per_team_extent = SourceTermToyProblem::getWorkSpaceSize(kmcd);

    Kokkos::parallel_for(
      profile_name,
      policy,
      KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
        const ordinal_type i = member.league_rank();
        const real_type_1d_view_type state_at_i =
          Kokkos::subview(state, i, Kokkos::ALL());
        const real_type_1d_view_type SourceTermToyProblem_at_i =
          Kokkos::subview(SourceTermToyProblem, i, Kokkos::ALL());

        Scratch<real_type_1d_view_type> work(member.team_scratch(level),
                                        per_team_extent);

        //
        const real_type theta_at_i = theta(i);
        const real_type lambda_at_i = lambda(i);

        Impl::SourceTermToyProblem<real_type, device_type>::team_invoke(member, theta_at_i, lambda_at_i,
          state_at_i, SourceTermToyProblem_at_i, work, kmcd);

      });
    Kokkos::Profiling::popRegion();
  }



void
SourceTermToyProblem::runDeviceBatch( /// input
  typename UseThisTeamPolicy<exec_space>::type& policy,
  const real_type_1d_view_type& theta,
  const real_type_1d_view_type& lambda,
  const real_type_2d_view_type& state,
  /// output
  const real_type_2d_view_type& SourceTermToyProblem,
  /// const data from kinetic model
  const kinetic_model_type& kmcd)
{

  SourceTermToyProblem_TemplateRun( /// input
    "TChem::SourceTermToyProblem::runDeviceBatch",
    /// team size setting
    policy,
    theta,
    lambda,
    state,
    SourceTermToyProblem,
    kmcd);

}

void
SourceTermToyProblem::runHostBatch( /// input
  typename UseThisTeamPolicy<host_exec_space>::type& policy,
  const real_type_1d_view_host_type& theta,
  const real_type_1d_view_host_type& lambda,
  const real_type_2d_view_host_type& state,
  /// output
  const real_type_2d_view_host_type& SourceTermToyProblem,
  /// const data from kinetic model
  const kinetic_model_host_type& kmcd)
{

  SourceTermToyProblem_TemplateRun( /// input
    "TChem::SourceTermToyProblem::runHostBatch",
    /// team size setting
    policy,
    theta,
    lambda,
    state,
    SourceTermToyProblem,
    kmcd);

}


} // namespace TChem
