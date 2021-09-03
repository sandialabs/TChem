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
#include "TChem_NetProductionRateSurfacePerMass.hpp"
#include "TChem_Impl_RhoMixMs.hpp"

namespace TChem {

void
NetProductionRateSurfacePerMass::runHostBatch( /// input
  const ordinal_type nBatch,
  const real_type_2d_view_host_type& state,
  /// input
  const real_type_2d_view_host_type& site_fraction,
  /// output
  const real_type_2d_view_host_type& omega,
  const real_type_2d_view_host_type& omegaSurf,
  /// const data from kinetic model
  const kinetic_model_host_type& kmcd,
  /// const data from kinetic model
  const kinetic_surf_model_host_type& kmcdSurf)
{
  Kokkos::Profiling::pushRegion("TChem::NetProductionRateSurfacePerMass::runHostBatch");
  using policy_type = Kokkos::TeamPolicy<host_exec_space>;
  using ReactionRatesSurface = Impl::ReactionRatesSurface<real_type,host_device_type>;

  const ordinal_type level = 1;
  const ordinal_type per_team_extent = getWorkSpaceSize(kmcd, kmcdSurf);
  const ordinal_type per_team_scratch =
    Scratch<real_type_1d_view_host_type>::shmem_size(per_team_extent);

  // policy_type policy(nBatch); // error
  policy_type policy(nBatch, Kokkos::AUTO()); // fine

  // policy_type policy(nBatch, Kokkos::AUTO(), Kokkos::AUTO()); // error
  policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
  Kokkos::parallel_for(
    "TChem::NetProductionRateSurfacePerMass::runHostBatch",
    policy,
    KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
      const ordinal_type i = member.league_rank();
      const real_type_1d_view_host_type state_at_i =
        Kokkos::subview(state, i, Kokkos::ALL());
      const real_type_1d_view_host_type omega_at_i =
        Kokkos::subview(omega, i, Kokkos::ALL());
      const real_type_1d_view_host_type omegaSurf_at_i =
        Kokkos::subview(omegaSurf, i, Kokkos::ALL());
      Scratch<real_type_1d_view_host_type> work(member.team_scratch(level),
                                           per_team_extent);
      // site fraction
      const real_type_1d_view_host_type Zk_at_i =
        Kokkos::subview(site_fraction, i, Kokkos::ALL());

      const Impl::StateVector<real_type_1d_view_host_type> sv_at_i(kmcd.nSpec,
                                                              state_at_i);
      TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                        "Error: input state vector is not valid");
      {
        const real_type t = sv_at_i.Temperature();
        const real_type p = sv_at_i.Pressure();
        const real_type_1d_view_host_type Ys = sv_at_i.MassFractions();
        // const real_type density = sv_at_i.Density();
        const real_type density = kmcd.rho < real_type(0)
                               ? Impl::RhoMixMs<real_type, host_device_type >::team_invoke(member, t, p, Ys, kmcd)
                               : kmcd.rho;
        //
        member.team_barrier();



        ReactionRatesSurface ::team_invoke(member,
                                                 t,
                                                 p,
                                                 density,
                                                 Ys,
                                                 Zk_at_i,
                                                 omega_at_i,
                                                 omegaSurf_at_i,
                                                 work,
                                                 kmcd,
                                                 kmcdSurf);
      }
    });
  Kokkos::Profiling::popRegion();
}

void
NetProductionRateSurfacePerMass::runDeviceBatch( /// input
  const ordinal_type nBatch,
  const real_type_2d_view_type& state,
  /// input
  const real_type_2d_view_type& site_fraction,
  /// output
  const real_type_2d_view_type& omega,
  const real_type_2d_view_type& omegaSurf,
  /// const data from kinetic model
  const kinetic_model_type& kmcd,
  /// const data from kinetic model surface
  const kinetic_surf_model_type& kmcdSurf)
{
  Kokkos::Profiling::pushRegion("TChem::NetProductionRateSurfacePerMass::runDeviceBatch");
  using policy_type = Kokkos::TeamPolicy<exec_space>;
  using ReactionRatesSurface = Impl::ReactionRatesSurface<real_type,device_type>;

  const ordinal_type level = 1;
  const ordinal_type per_team_extent = getWorkSpaceSize(kmcd, kmcdSurf);
  const ordinal_type per_team_scratch =
    Scratch<real_type_1d_view_type>::shmem_size(per_team_extent);

  // policy_type policy(nBatch); // error
  policy_type policy(nBatch, Kokkos::AUTO()); // fine
  // policy_type policy(nBatch, Kokkos::AUTO(), Kokkos::AUTO()); // error
  policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
  Kokkos::parallel_for(
    "TChem::NetProductionRateSurfacePerMass::runDeviceBatch",
    policy,
    KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
      const ordinal_type i = member.league_rank();
      const real_type_1d_view_type state_at_i =
        Kokkos::subview(state, i, Kokkos::ALL());
      const real_type_1d_view_type omega_at_i =
        Kokkos::subview(omega, i, Kokkos::ALL());
      const real_type_1d_view_type omegaSurf_at_i =
        Kokkos::subview(omegaSurf, i, Kokkos::ALL());
      // site fraction
      const real_type_1d_view_type Zk_at_i =
        Kokkos::subview(site_fraction, i, Kokkos::ALL());
      Scratch<real_type_1d_view_type> work(member.team_scratch(level),
                                      per_team_extent);

      const Impl::StateVector<real_type_1d_view_type> sv_at_i(kmcd.nSpec,
                                                         state_at_i);
      TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                        "Error: input state vector is not valid");
      {
        const real_type t = sv_at_i.Temperature();
        const real_type p = sv_at_i.Pressure();
        const real_type_1d_view_type Ys = sv_at_i.MassFractions();
        // const real_type density = sv_at_i.Density();
        const real_type density = kmcd.rho < real_type(0)
                               ? Impl::RhoMixMs<real_type, device_type >::team_invoke(member, t, p, Ys, kmcd)
                               : kmcd.rho;
        //
        member.team_barrier();


        ReactionRatesSurface ::team_invoke(member,
                                                 t,
                                                 p,
                                                 density,
                                                 Ys,
                                                 Zk_at_i,
                                                 omega_at_i,
                                                 omegaSurf_at_i,
                                                 work,
                                                 kmcd,
                                                 kmcdSurf);
      }
    });
  Kokkos::Profiling::popRegion();
}

} // namespace TChem
