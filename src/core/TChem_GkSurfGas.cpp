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

#include "TChem_GkSurfGas.hpp"
#include "TChem_Impl_Gk.hpp"

namespace TChem {

void
GkSurfGas::runHostBatch( /// input
  const ordinal_type nBatch,
  const real_type_2d_view_host& state,
  /// output
  const real_type_2d_view_host& gk,
  const real_type_2d_view_host& gkSurf,
  const real_type_2d_view_host& hks,
  const real_type_2d_view_host& hksSurf,
  /// const data from kinetic model
  const KineticModelConstDataHost& kmcd,
  /// const data from kinetic model
  const KineticSurfModelConstDataHost& kmcdSurf)
{
  Kokkos::Profiling::pushRegion("TChem::GkSurfGas::runHostBatch");
  using policy_type = Kokkos::TeamPolicy<host_exec_space>;

  const ordinal_type level = 1;
  const ordinal_type per_team_extent = getWorkSpaceSize(kmcd, kmcdSurf);
  const ordinal_type per_team_scratch =
    Scratch<real_type_1d_view_host>::shmem_size(per_team_extent);

  // policy_type policy(nBatch); // error
  policy_type policy(nBatch, Kokkos::AUTO()); // fine

  // policy_type policy(nBatch, Kokkos::AUTO(), Kokkos::AUTO()); // error
  policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
  Kokkos::parallel_for(
    "TChem::GkSurfGas::runHostBatch",
    policy,
    KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
      const ordinal_type i = member.league_rank();
      const real_type_1d_view_host state_at_i =
        Kokkos::subview(state, i, Kokkos::ALL());

      const real_type_1d_view_host gk_at_i =
        Kokkos::subview(gk, i, Kokkos::ALL());
      const real_type_1d_view_host gkSurf_at_i =
        Kokkos::subview(gkSurf, i, Kokkos::ALL());

      const real_type_1d_view_host hks_at_i =
        Kokkos::subview(hks, i, Kokkos::ALL());
      const real_type_1d_view_host hksSurf_at_i =
        Kokkos::subview(hksSurf, i, Kokkos::ALL());

      Scratch<real_type_1d_view_host> work(member.team_scratch(level),
                                           per_team_extent);

      const Impl::StateVector<real_type_1d_view_host> sv_at_i(kmcd.nSpec,
                                                              state_at_i);
      auto w = (real_type*)work.data();
      auto cpks = real_type_1d_view_host(w, kmcd.nSpec);
      w += kmcd.nSpec;
      auto cpksSurf = real_type_1d_view_host(w, kmcdSurf.nSpec);
      w += kmcdSurf.nSpec;

      TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                        "Error: input state vector is not valid");
      {
        const real_type t = sv_at_i.Temperature();
        const real_type p = sv_at_i.Pressure();
        const real_type_1d_view_host Xc = sv_at_i.MassFractions();

        // gas
        Impl::GkSurfGas ::team_invoke(member,
                                      t, /// input
                                      gk_at_i,
                                      hks_at_i, /// output
                                      cpks,     /// workspace
                                      kmcd);

        // surfaces
        Impl::GkSurfGas ::team_invoke(member,
                                      t, /// input
                                      gkSurf_at_i,
                                      hksSurf_at_i, /// output
                                      cpksSurf,     /// workspace
                                      kmcdSurf);
      }
    });
  Kokkos::Profiling::popRegion();
}

void
GkSurfGas::runDeviceBatch( /// input
  const ordinal_type nBatch,
  const real_type_2d_view& state,
  /// output
  const real_type_2d_view& gk,
  const real_type_2d_view& gkSurf,
  const real_type_2d_view& hks,
  const real_type_2d_view& hksSurf,
  /// const data from kinetic model
  const KineticModelConstDataDevice& kmcd,
  /// const data from kinetic model surface
  const KineticSurfModelConstDataDevice& kmcdSurf)
{
  Kokkos::Profiling::pushRegion("TChem::GkSurfGas::runDeviceBatch");
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
    "TChem::GkSurfGas::runDeviceBatch",
    policy,
    KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
      const ordinal_type i = member.league_rank();
      const real_type_1d_view state_at_i =
        Kokkos::subview(state, i, Kokkos::ALL());

      const real_type_1d_view gk_at_i = Kokkos::subview(gk, i, Kokkos::ALL());
      const real_type_1d_view gkSurf_at_i =
        Kokkos::subview(gkSurf, i, Kokkos::ALL());

      const real_type_1d_view hks_at_i = Kokkos::subview(hks, i, Kokkos::ALL());
      const real_type_1d_view hksSurf_at_i =
        Kokkos::subview(hksSurf, i, Kokkos::ALL());

      Scratch<real_type_1d_view> work(member.team_scratch(level),
                                      per_team_extent);

      auto w = (real_type*)work.data();
      auto cpks = real_type_1d_view(w, kmcd.nSpec);
      w += kmcd.nSpec;
      auto cpksSurf = real_type_1d_view(w, kmcdSurf.nSpec);
      w += kmcdSurf.nSpec;

      const Impl::StateVector<real_type_1d_view> sv_at_i(kmcd.nSpec,
                                                         state_at_i);
      TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                        "Error: input state vector is not valid");
      {
        const real_type t = sv_at_i.Temperature();
        const real_type p = sv_at_i.Pressure();
        const real_type_1d_view Xc = sv_at_i.MassFractions();

        // gas
        Impl::GkSurfGas ::team_invoke(member,
                                      t, /// input
                                      gk_at_i,
                                      hks_at_i, /// output
                                      cpks,     /// workspace
                                      kmcd);

        // surfaces
        Impl::GkSurfGas ::team_invoke(member,
                                      t, /// input
                                      gkSurf_at_i,
                                      hksSurf_at_i, /// output
                                      cpksSurf,     /// workspace
                                      kmcdSurf);
      }
    });

  Kokkos::Profiling::popRegion();
}

} // namespace TChem
