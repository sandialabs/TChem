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
#include "TChem_Util.hpp"

#include "TChem_Impl_ReactionRatesSurface.hpp"
#include "TChem_NetProductionRateSurfacePerMole.hpp"

namespace TChem {

void
NetProductionRateSurfacePerMole::runHostBatch( /// input
  const ordinal_type nBatch,
  const real_type_2d_view_host& state,
  /// input
  const real_type_2d_view_host& zSurf,
  /// output
  const real_type_2d_view_host& omega,
  const real_type_2d_view_host& omegaSurf,
  /// const data from kinetic model
  const KineticModelConstDataHost& kmcd,
  /// const data from kinetic model
  const KineticSurfModelConstDataHost& kmcdSurf)
{
  Kokkos::Profiling::pushRegion("TChem::NetProductionRateSurfacePerMole::runHostBatch");
  using policy_type = Kokkos::TeamPolicy<host_exec_space>;

  const ordinal_type level = 1;
  const ordinal_type per_team_extent = getWorkSpaceSize(kmcd, kmcdSurf);
  const ordinal_type per_team_scratch =
    Scratch<real_type_1d_view>::shmem_size(per_team_extent);

  // policy_type policy(nBatch); // error
  policy_type policy(nBatch, Kokkos::AUTO()); // fine

  // policy_type policy(nBatch, Kokkos::AUTO(), Kokkos::AUTO()); // error
  policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
  Kokkos::parallel_for(
    "TChem::NetProductionRateSurfacePerMole::runHostBatch",
    policy,
    KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
      const ordinal_type i = member.league_rank();
      const real_type_1d_view_host state_at_i =
        Kokkos::subview(state, i, Kokkos::ALL());
      const real_type_1d_view_host omega_at_i =
        Kokkos::subview(omega, i, Kokkos::ALL());
      const real_type_1d_view_host omegaSurf_at_i =
        Kokkos::subview(omegaSurf, i, Kokkos::ALL());
      Scratch<real_type_1d_view_host> work(member.team_scratch(level),
                                           per_team_extent);
      // site fraction
      const real_type_1d_view_host Zk_at_i =
        Kokkos::subview(zSurf, i, Kokkos::ALL());

      const Impl::StateVector<real_type_1d_view_host> sv_at_i(kmcd.nSpec,
                                                              state_at_i);
      TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                        "Error: input state vector is not valid");
      {
        const real_type t = sv_at_i.Temperature();
        const real_type p = sv_at_i.Pressure();
        const real_type_1d_view_host Xc = sv_at_i.MassFractions();

        Impl::ReactionRatesSurface ::team_invoke(member,
                                                 t,
                                                 p,
                                                 Xc,
                                                 Zk_at_i,
                                                 omega_at_i,
                                                 omegaSurf_at_i,
                                                 work,
                                                 kmcd,
                                                 kmcdSurf);

        //
        //
        member.team_barrier();

        //
        Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, kmcd.nSpec),
          [&](const ordinal_type& k) {
            omega_at_i(k) /=kmcd.sMass(k);
          });
        //
        Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, kmcdSurf.nSpec),
          [&](const ordinal_type& k) {
            omegaSurf_at_i(k) /=kmcd.sMass(k);
          });                                         
      }
    });
  Kokkos::Profiling::popRegion();
}

void
NetProductionRateSurfacePerMole::runDeviceBatch( /// input
  const ordinal_type nBatch,
  const real_type_2d_view& state,
  /// input
  const real_type_2d_view& zSurf,
  /// output
  const real_type_2d_view& omega,
  const real_type_2d_view& omegaSurf,
  /// const data from kinetic model
  const KineticModelConstDataDevice& kmcd,
  /// const data from kinetic model surface
  const KineticSurfModelConstDataDevice& kmcdSurf)
{
  Kokkos::Profiling::pushRegion("TChem::NetProductionRateSurfacePerMole::runDeviceBatch");
  using policy_type = Kokkos::TeamPolicy<exec_space>;

  const ordinal_type level = 1;
  const ordinal_type per_team_extent = getWorkSpaceSize(kmcd, kmcdSurf);
  const ordinal_type per_team_scratch =
    Scratch<real_type_1d_view>::shmem_size(per_team_extent);

  // policy_type policy(nBatch); // error
  policy_type policy(nBatch, Kokkos::AUTO()); // fine
  // policy_type policy(nBatch, Kokkos::AUTO(), Kokkos::AUTO()); // error
  policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
  Kokkos::parallel_for(
    "TChem::NetProductionRateSurfacePerMole::runDeviceBatch",
    policy,
    KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
      const ordinal_type i = member.league_rank();
      const real_type_1d_view state_at_i =
        Kokkos::subview(state, i, Kokkos::ALL());
      const real_type_1d_view omega_at_i =
        Kokkos::subview(omega, i, Kokkos::ALL());
      const real_type_1d_view omegaSurf_at_i =
        Kokkos::subview(omegaSurf, i, Kokkos::ALL());
      // site fraction
      const real_type_1d_view Zk_at_i =
        Kokkos::subview(zSurf, i, Kokkos::ALL());
      Scratch<real_type_1d_view> work(member.team_scratch(level),
                                      per_team_extent);

      const Impl::StateVector<real_type_1d_view> sv_at_i(kmcd.nSpec,
                                                         state_at_i);
      TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                        "Error: input state vector is not valid");
      {
        const real_type t = sv_at_i.Temperature();
        const real_type p = sv_at_i.Pressure();
        const real_type_1d_view Xc = sv_at_i.MassFractions();

        Impl::ReactionRatesSurface ::team_invoke(member,
                                                 t,
                                                 p,
                                                 Xc,
                                                 Zk_at_i,
                                                 omega_at_i,
                                                 omegaSurf_at_i,
                                                 work,
                                                 kmcd,
                                                 kmcdSurf);


        //
        member.team_barrier();

        //
        Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, kmcd.nSpec),
          [&](const ordinal_type& k) {
            omega_at_i(k) /=kmcd.sMass(k);
          });
        //
        Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, kmcdSurf.nSpec),
          [&](const ordinal_type& k) {
            omegaSurf_at_i(k) /=kmcd.sMass(k);
          });

      }
    });
  Kokkos::Profiling::popRegion();
}

} // namespace TChem
