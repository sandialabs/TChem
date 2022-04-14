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

#include "TChem_Impl_ReactionRates.hpp"

#include "TChem_NetProductionRatePerMole.hpp"

namespace TChem {

void
NetProductionRatePerMole::runHostBatch( /// input
  const ordinal_type nBatch,
  const real_type_2d_view_host_type& state,
  /// output
  const real_type_2d_view_host_type& omega,
  /// const data from kinetic model
  const kinetic_model_host_type& kmcd)
{
  Kokkos::Profiling::pushRegion("TChem::NetProductionRatePerMole::runHostBatch");
  using policy_type = Kokkos::TeamPolicy<host_exec_space>;

  const ordinal_type level = 1;
  const ordinal_type per_team_extent = getWorkSpaceSize(kmcd);
  const ordinal_type per_team_scratch =
    Scratch<real_type_1d_view_host_type>::shmem_size(per_team_extent);

  // policy_type policy(nBatch); // error
  policy_type policy(nBatch, Kokkos::AUTO()); // fine
  // policy_type policy(nBatch, Kokkos::AUTO(), Kokkos::AUTO()); // error
  policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
  Kokkos::parallel_for(
    "TChem::NetProductionRatePerMole::runHostBatch",
    policy,
    KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
      const ordinal_type i = member.league_rank();
      const real_type_1d_view_host_type state_at_i =
        Kokkos::subview(state, i, Kokkos::ALL());
      const real_type_1d_view_host_type omega_at_i =
        Kokkos::subview(omega, i, Kokkos::ALL());
      Scratch<real_type_1d_view_host_type> work(member.team_scratch(level),
                                           per_team_extent);

      const Impl::StateVector<real_type_1d_view_host_type> sv_at_i(kmcd.nSpec,
                                                              state_at_i);
      TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                        "Error: input state vector is not valid");
      {
        const real_type t = sv_at_i.Temperature();
        const real_type p = sv_at_i.Pressure();
        const real_type density = sv_at_i.Density();

        auto w = work.data();
        auto ww = real_type_1d_view_host_type(w, work.span());

        const real_type_1d_view_host_type Xc = sv_at_i.MassFractions();
        Impl::ReactionRates<real_type, host_device_type >::team_invoke(
          member, t, p, density, Xc, omega_at_i, ww, kmcd);
      }
    });
  Kokkos::Profiling::popRegion();
}

void
NetProductionRatePerMole::runDeviceBatch( /// input
  const ordinal_type nBatch,
  const real_type_2d_view_type& state,
  /// output
  const real_type_2d_view_type& omega,
  /// const data from kinetic model
  const kinetic_model_type& kmcd)
{
  Kokkos::Profiling::pushRegion("TChem::NetProductionRatePerMole::runDeviceBatch");
  using policy_type = Kokkos::TeamPolicy<exec_space>;

  const ordinal_type level = 1;
  const ordinal_type per_team_extent = getWorkSpaceSize(kmcd);
  const ordinal_type per_team_scratch =
    Scratch<real_type_1d_view_type>::shmem_size(per_team_extent);

  // policy_type policy(nBatch); // error
  policy_type policy(nBatch, Kokkos::AUTO()); // fine
  // policy_type policy(nBatch, Kokkos::AUTO(), Kokkos::AUTO()); // error
  policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
  Kokkos::parallel_for(
    "TChem::NetProductionRatePerMole::runDeviceBatch",
    policy,
    KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
      const ordinal_type i = member.league_rank();
      const real_type_1d_view_type state_at_i =
        Kokkos::subview(state, i, Kokkos::ALL());
      const real_type_1d_view_type omega_at_i =
        Kokkos::subview(omega, i, Kokkos::ALL());
      Scratch<real_type_1d_view_type> work(member.team_scratch(level),
                                      per_team_extent);

      const Impl::StateVector<real_type_1d_view_type> sv_at_i(kmcd.nSpec,
                                                         state_at_i);
      TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                        "Error: input state vector is not valid");
      {
        const real_type t = sv_at_i.Temperature();
        const real_type p = sv_at_i.Pressure();
        const real_type density = sv_at_i.Density();
        const real_type_1d_view_type Xc = sv_at_i.MassFractions();
        auto w = work.data();
        auto ww = real_type_1d_view_type(w, work.span());

        Impl::ReactionRates<real_type, device_type >::team_invoke(
          member, t, p, density, Xc, omega_at_i, ww, kmcd);

        member.team_barrier();

        Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, kmcd.nSpec),
          [&](const ordinal_type& k) {
            omega_at_i(k) /=kmcd.sMass(k);
          });
      }
    });
  Kokkos::Profiling::popRegion();
}

} // namespace TChem
