/* =====================================================================================
TChem version 2.1.0
Copyright (2020) NTESS
https://github.com/sandialabs/TChem

Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains 
certain rights in this software.

This file is part of TChem. TChem is open-source software: you can redistribute it
and/or modify it under the terms of BSD 2-Clause License
(https://opensource.org/licenses/BSD-2-Clause). A copy of the license is also
provided under the main directory

Questions? Contact Cosmin Safta at <csafta@sandia.gov>, or
           Kyungjoo Kim at <kyukim@sandia.gov>, or
           Oscar Diaz-Ibarra at <odiazib@sandia.gov>

Sandia National Laboratories, Livermore, CA, USA
===================================================================================== */


#include "TChem_SourceTerm.hpp"

namespace TChem {

  template<typename PolicyType,
           typename RealType1DViewType,
           typename RealType2DViewType,
           typename KineticModelConstType>
void
SourceTerm_TemplateRun( /// input
  const std::string& profile_name,
  const RealType1DViewType& dummy_1d,
  /// team size setting
  const PolicyType& policy,

  const RealType2DViewType& state,
  /// output
  const RealType2DViewType& SourceTerm,
  /// const data from kinetic model
  const KineticModelConstType& kmcd)
{
  Kokkos::Profiling::pushRegion(profile_name);
  using policy_type = PolicyType;
  const ordinal_type level = 1;
  const ordinal_type per_team_extent = TChem::SourceTerm::getWorkSpaceSize(kmcd);

  Kokkos::parallel_for(
    profile_name,
    policy,
    KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
      const ordinal_type i = member.league_rank();
      const RealType1DViewType state_at_i =
        Kokkos::subview(state, i, Kokkos::ALL());
      const RealType1DViewType SourceTerm_at_i =
        Kokkos::subview(SourceTerm, i, Kokkos::ALL());

      Scratch<RealType1DViewType> work(member.team_scratch(level),
                                      per_team_extent);

      const Impl::StateVector<RealType1DViewType> sv_at_i(kmcd.nSpec,
                                                         state_at_i);
      TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                        "Error: input state vector is not valid");
      {
        const auto t = sv_at_i.Temperature();
        const auto p = sv_at_i.Pressure();
        const RealType1DViewType Ys = sv_at_i.MassFractions();

        Impl::SourceTerm::team_invoke(
          member, t, p, Ys, SourceTerm_at_i, work, kmcd);
      }
    });
  Kokkos::Profiling::popRegion();
}

void
SourceTerm::runDeviceBatch( /// thread block size
  typename UseThisTeamPolicy<exec_space>::type& policy,
  const real_type_2d_view& state,
  const real_type_2d_view& SourceTerm,
  const KineticModelConstDataDevice& kmcd
)
{
  SourceTerm_TemplateRun(
  "TChem::SourceTerm::runDeviceBatch",
  real_type_1d_view(),
  /// team policy
  policy,
  state,
  SourceTerm,
  kmcd);
}

void
SourceTerm::runDeviceBatch( /// thread block size
  const ordinal_type nBatch,
  const real_type_2d_view& state,
  const real_type_2d_view& SourceTerm,
  const KineticModelConstDataDevice& kmcd)
{
  // setting policy
  const auto exec_space_instance = TChem::exec_space();
  using policy_type =
    typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;
  const ordinal_type level = 1;
  const ordinal_type per_team_extent =
   TChem::SourceTerm
        ::getWorkSpaceSize(kmcd); ///
  const ordinal_type per_team_scratch =
      Scratch<real_type_1d_view>::shmem_size(per_team_extent);
  policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());

  policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

  SourceTerm_TemplateRun(
  "TChem::SourceTerm::runDeviceBatch",
  real_type_1d_view(),
  /// team policy
  policy,
  state,
  SourceTerm,
  kmcd);
}


void
SourceTerm::runHostBatch( /// thread block size
  const ordinal_type nBatch,
  const real_type_2d_view_host& state,
  const real_type_2d_view_host& SourceTerm,
  const KineticModelConstDataHost& kmcd)
{
  // setting policy
  const auto exec_space_instance = TChem::host_exec_space();
  using policy_type =
    typename TChem::UseThisTeamPolicy<TChem::host_exec_space>::type;
  const ordinal_type level = 1;
  const ordinal_type per_team_extent =
   TChem::SourceTerm
        ::getWorkSpaceSize(kmcd); ///
  const ordinal_type per_team_scratch =
      Scratch<real_type_1d_view_host>::shmem_size(per_team_extent);
  policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());

  policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

  SourceTerm_TemplateRun(
  "TChem::SourceTerm::runDeviceBatch",
  real_type_1d_view_host(),
  /// team policy
  policy,
  state,
  SourceTerm,
  kmcd);
}

void
SourceTerm::runHostBatch( /// thread block size
  typename UseThisTeamPolicy<host_exec_space>::type& policy,
  const real_type_2d_view_host& state,
  const real_type_2d_view_host& SourceTerm,
  const KineticModelConstDataHost& kmcd
)
{
  SourceTerm_TemplateRun(
  "TChem::SourceTerm::runHostBatch",
  real_type_1d_view_host(),
  /// team policy
  policy,
  state,
  SourceTerm,
  kmcd);
}


} // namespace TChem
