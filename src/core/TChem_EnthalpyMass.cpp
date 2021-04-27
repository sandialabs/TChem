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
#include "TChem_EnthalpyMass.hpp"

namespace TChem {

namespace Impl {
template<typename PolicyType,
         typename RealType0DViewType,
         typename RealType1DViewType,
         typename RealType2DViewType,
         typename KineticModelConstType>
void
EnthalpyMass_TemplateRun( /// required template arguments
  const std::string& profile_name,
  const RealType0DViewType& dummy_0d,
  /// team size setting
  const PolicyType& policy,
  const RealType2DViewType& state,
  // outputFile
  const RealType2DViewType& EnthalpyMass,
  const RealType1DViewType& EnthalpyMixMass,
  const KineticModelConstType& kmcd)
{
  Kokkos::Profiling::pushRegion(profile_name);
  using policy_type = PolicyType;
  const ordinal_type level = 1; ///
  const ordinal_type per_team_extent = EnthalpyMass::getWorkSpaceSize(kmcd);

  Kokkos::parallel_for(
    profile_name,
    policy,
    KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
      const ordinal_type i = member.league_rank();
      const RealType1DViewType state_at_i =
        Kokkos::subview(state, i, Kokkos::ALL());
      const RealType1DViewType EnthalpyMass_at_i =
        Kokkos::subview(EnthalpyMass, i, Kokkos::ALL());
      const RealType0DViewType EnthalpyMixMass_at_i =
        Kokkos::subview(EnthalpyMixMass, i);
      Scratch<RealType1DViewType> work(member.team_scratch(level),
                                       per_team_extent);
      auto w = (real_type*)work.data();
      auto cpks = RealType1DViewType(w, kmcd.nSpec);
      w += kmcd.nSpec;

      const Impl::StateVector<RealType1DViewType> sv_at_i(kmcd.nSpec,
                                                          state_at_i);
      TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                        "Error: input state vector is not valid");
      {
        const real_type t = sv_at_i.Temperature();
        const real_type p = sv_at_i.Pressure();
        const RealType1DViewType Ys = sv_at_i.MassFractions();

        Impl::EnthalpySpecMs ::team_invoke(
          member, t, EnthalpyMass_at_i, cpks, kmcd);

        member.team_barrier();

        //
        real_type sumHk(0);
        Kokkos::parallel_reduce(
          Kokkos::TeamVectorRange(member, kmcd.nSpec),
          [&](const ordinal_type& k, real_type& update) {
            update += EnthalpyMass_at_i(k) * Ys(k);
          },
          sumHk);

        EnthalpyMixMass_at_i() = sumHk;
      }
    });
  Kokkos::Profiling::popRegion();
}

} // namespace Impl

void
EnthalpyMass::runDeviceBatch( /// thread block size
  typename UseThisTeamPolicy<exec_space>::type& policy,
  const real_type_2d_view& state,
  /// output
  const real_type_2d_view& EnthalpyMass,
  const real_type_1d_view& EnthalpyMixMass,
  /// const data from kinetic model
  const KineticModelConstDataDevice& kmcd)
{
  Impl::EnthalpyMass_TemplateRun("TChem::EnthalpyMass::runDeviceBatch",
                                 real_type_0d_view(),
                                 /// team policy
                                 policy,
                                 /// input
                                 state,
                                 EnthalpyMass,
                                 EnthalpyMixMass,
                                 kmcd);
}

void
EnthalpyMass::runDeviceBatch( /// thread block size
  const exec_space& exec_space_instance,
  const ordinal_type& team_size,
  const ordinal_type& vector_size,
  const ordinal_type nBatch,
  const real_type_2d_view& state,
  /// output
  const real_type_2d_view& EnthalpyMass,
  const real_type_1d_view& EnthalpyMixMass,
  /// const data from kinetic model
  const KineticModelConstDataDevice& kmcd)
{
  // setting policy
  using policy_type =
    typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;
  const ordinal_type level = 1;
  const ordinal_type per_team_extent =

  TChem::EnthalpyMass::getWorkSpaceSize(kmcd); ///
  const ordinal_type per_team_scratch =
      Scratch<real_type_1d_view>::shmem_size(per_team_extent);
  policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());

  if (team_size > 0 && vector_size > 0) {
    policy = policy_type(exec_space_instance, nBatch, team_size, vector_size);
  }

  policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));


  Impl::EnthalpyMass_TemplateRun("TChem::EnthalpyMass::runDeviceBatch",
                                 real_type_0d_view(),
                                 /// team policy
                                 policy,
                                 /// input
                                 state,
                                 EnthalpyMass,
                                 EnthalpyMixMass,
                                 kmcd);
}

void
EnthalpyMass::runHostBatch( /// thread block size
  typename UseThisTeamPolicy<host_exec_space>::type& policy,
  const real_type_2d_view_host& state,
  /// output
  const real_type_2d_view_host& EnthalpyMass,
  const real_type_1d_view_host& EnthalpyMixMass,
  /// const data from kinetic model
  const KineticModelConstDataHost& kmcd)
{
  Impl::EnthalpyMass_TemplateRun("TChem::EnthalpyMass::runHostBatch",
                                 real_type_0d_view_host(),
                                 /// team policy
                                 policy,
                                 /// input
                                 state,
                                 EnthalpyMass,
                                 EnthalpyMixMass,
                                 kmcd);
}

} // namespace TChem
