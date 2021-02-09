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



#include "TChem_RateOfProgress.hpp"

namespace TChem {
  template<typename PolicyType,
           typename RealType1DViewType,
           typename RealType2DViewType,
           typename KineticModelConstType>

  //
  void
  RateOfProgress_TemplateRun( /// input
    const std::string& profile_name,
    const RealType1DViewType& dummy_1d,
    /// team size setting
    const PolicyType& policy,
    const RealType2DViewType& state,
    const RealType2DViewType& RoPFor,
    const RealType2DViewType& RoPRev,
    const KineticModelConstType& kmcd
  )
  {
    Kokkos::Profiling::pushRegion(profile_name);
    using policy_type = PolicyType;

    const ordinal_type level = 1;
    const ordinal_type per_team_extent = RateOfProgress::getWorkSpaceSize(kmcd);

    Kokkos::parallel_for(
      profile_name,
      policy,
      KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
        const ordinal_type i = member.league_rank();
        const RealType1DViewType state_at_i =
          Kokkos::subview(state, i, Kokkos::ALL());
        const RealType1DViewType RoPFor_at_i =
          Kokkos::subview(RoPFor, i, Kokkos::ALL());
        const RealType1DViewType RoPRev_at_i =
          Kokkos::subview(RoPRev, i, Kokkos::ALL());

        Scratch<RealType1DViewType> work(member.team_scratch(level),
                                        per_team_extent);

        const Impl::StateVector<RealType1DViewType> sv_at_i(kmcd.nSpec,
                                                           state_at_i);
        TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                          "Error: input state vector is not valid");
        {
          const real_type t = sv_at_i.Temperature();
          const real_type p = sv_at_i.Pressure();
          const RealType1DViewType Ys = sv_at_i.MassFractions();

          Impl::RateOfProgressInd ::team_invoke(
            member, t, p, Ys, RoPFor_at_i, RoPRev_at_i, work, kmcd);
        }
      });
    Kokkos::Profiling::popRegion();
  }


void
RateOfProgress::runDeviceBatch( /// input
  const ordinal_type nBatch,
  const real_type_2d_view& state,
  /// output
  const real_type_2d_view& RoPFor,
  const real_type_2d_view& RoPRev,
  /// const data from kinetic model
  const KineticModelConstDataDevice& kmcd)
{

  using policy_type = Kokkos::TeamPolicy<exec_space>;
  const ordinal_type level = 1;
  const ordinal_type per_team_extent = RateOfProgress::getWorkSpaceSize(kmcd);
  const ordinal_type per_team_scratch =
    Scratch<real_type_1d_view>::shmem_size(per_team_extent);

  // policy_type policy(nBatch); // error
  policy_type policy(nBatch, Kokkos::AUTO()); // fine
  // policy_type policy(nBatch, Kokkos::AUTO(), Kokkos::AUTO()); // error
  policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

  RateOfProgress_TemplateRun( /// input
    "TChem::RateOfProgress::runDeviceBatch",
    real_type_1d_view(),
    /// team size setting
    policy,
    state,
    RoPFor,
    RoPRev,
    kmcd);
}


void
RateOfProgress::runDeviceBatch( /// input
  typename UseThisTeamPolicy<exec_space>::type& policy,
  const real_type_2d_view& state,
  /// output
  const real_type_2d_view& RoPFor,
  const real_type_2d_view& RoPRev,
  /// const data from kinetic model
  const KineticModelConstDataDevice& kmcd)
{

  RateOfProgress_TemplateRun( /// input
    "TChem::RateOfProgress::runDeviceBatch",
    real_type_1d_view(),
    /// team size setting
    policy,
    state,
    RoPFor,
    RoPRev,
    kmcd);
}

void
RateOfProgress::runHostBatch( /// input
  typename UseThisTeamPolicy<host_exec_space>::type& policy,
  const real_type_2d_view_host& state,
  /// output
  const real_type_2d_view_host& RoPFor,
  const real_type_2d_view_host& RoPRev,
  /// const data from kinetic model
  const KineticModelConstDataHost& kmcd)
  {

    RateOfProgress_TemplateRun( /// input
      "TChem::RateOfProgress::runHostBatch",
      real_type_1d_view_host(),
      /// team size setting
      policy,
      state,
      RoPFor,
      RoPRev,
      kmcd);
  }

  void
  RateOfProgress::runHostBatch( /// input
    const ordinal_type nBatch,
    const real_type_2d_view_host& state,
    /// output
    const real_type_2d_view_host& RoPFor,
    const real_type_2d_view_host& RoPRev,
    /// const data from kinetic model
    const KineticModelConstDataHost& kmcd)
  {

    using policy_type = Kokkos::TeamPolicy<host_exec_space>;
    const ordinal_type level = 1;
    const ordinal_type per_team_extent = RateOfProgress::getWorkSpaceSize(kmcd);
    const ordinal_type per_team_scratch =
      Scratch<real_type_1d_view_host>::shmem_size(per_team_extent);

    // policy_type policy(nBatch); // error
    policy_type policy(nBatch, Kokkos::AUTO()); // fine
    // policy_type policy(nBatch, Kokkos::AUTO(), Kokkos::AUTO()); // error
    policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

    RateOfProgress_TemplateRun( /// input
      "TChem::RateOfProgress::runHostBatch",
      real_type_1d_view_host(),
      /// team size setting
      policy,
      state,
      RoPFor,
      RoPRev,
      kmcd);
  }




} // namespace TChem
