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
  const real_type_2d_view_host_type& state,
  /// output
  const real_type_2d_view_host_type& gk,
  const real_type_2d_view_host_type& gkSurf,
  const real_type_2d_view_host_type& hks,
  const real_type_2d_view_host_type& hksSurf,
  /// const data from kinetic model
  const kinetic_model_host_type& kmcd,
  /// const data from kinetic model
  const kinetic_surf_model_host_type& kmcdSurf)
{
  Kokkos::Profiling::pushRegion("TChem::GkSurfGas::runHostBatch");
  using policy_type = Kokkos::TeamPolicy<host_exec_space>;
  using GkSurfGas = Impl::GkFcnSurfGas<real_type, host_device_type>;

  const ordinal_type level = 1;
  const ordinal_type per_team_extent = getWorkSpaceSize(kmcd, kmcdSurf);
  const ordinal_type per_team_scratch =
    Scratch<real_type_1d_view_host_type>::shmem_size(per_team_extent);

  // policy_type policy(nBatch); // error
  policy_type policy(nBatch, Kokkos::AUTO()); // fine

  // policy_type policy(nBatch, Kokkos::AUTO(), Kokkos::AUTO()); // error
  policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
  Kokkos::parallel_for(
    "TChem::GkSurfGas::runHostBatch",
    policy,
    KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
      const ordinal_type i = member.league_rank();
      const real_type_1d_view_host_type state_at_i =
        Kokkos::subview(state, i, Kokkos::ALL());

      const real_type_1d_view_host_type gk_at_i =
        Kokkos::subview(gk, i, Kokkos::ALL());
      const real_type_1d_view_host_type gkSurf_at_i =
        Kokkos::subview(gkSurf, i, Kokkos::ALL());

      const real_type_1d_view_host_type hks_at_i =
        Kokkos::subview(hks, i, Kokkos::ALL());
      const real_type_1d_view_host_type hksSurf_at_i =
        Kokkos::subview(hksSurf, i, Kokkos::ALL());

      Scratch<real_type_1d_view_host_type> work(member.team_scratch(level),
                                           per_team_extent);

      const Impl::StateVector<real_type_1d_view_host_type> sv_at_i(kmcd.nSpec,
                                                              state_at_i);
      auto w = (real_type*)work.data();
      auto cpks = real_type_1d_view_host_type(w, kmcd.nSpec);
      w += kmcd.nSpec;
      auto cpksSurf = real_type_1d_view_host_type(w, kmcdSurf.nSpec);
      w += kmcdSurf.nSpec;

      TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                        "Error: input state vector is not valid");
      {
        const real_type t = sv_at_i.Temperature();
        const real_type p = sv_at_i.Pressure();
        const real_type_1d_view_host_type Xc = sv_at_i.MassFractions();

        // gas
        GkSurfGas ::team_invoke(member,
                                      t, /// input
                                      gk_at_i,
                                      hks_at_i, /// output
                                      cpks,     /// workspace
                                      kmcd);

        // surfaces
        GkSurfGas ::team_invoke(member,
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
  const real_type_2d_view_type& state,
  /// output
  const real_type_2d_view_type& gk,
  const real_type_2d_view_type& gkSurf,
  const real_type_2d_view_type& hks,
  const real_type_2d_view_type& hksSurf,
  /// const data from kinetic model
  const kinetic_model_type& kmcd,
  /// const data from kinetic model surface
  const kinetic_surf_model_type& kmcdSurf)
{
  Kokkos::Profiling::pushRegion("TChem::GkSurfGas::runDeviceBatch");
  using policy_type = Kokkos::TeamPolicy<exec_space>;
  using GkSurfGas = Impl::GkFcnSurfGas<real_type,device_type>;


  const ordinal_type level = 1;
  const ordinal_type per_team_extent = getWorkSpaceSize(kmcd, kmcdSurf);
  const ordinal_type per_team_scratch =
    Scratch<real_type_1d_view_type>::shmem_size(per_team_extent);

  // policy_type policy(nBatch); // error
  policy_type policy(nBatch, Kokkos::AUTO()); // fine
  // policy_type policy(nBatch, Kokkos::AUTO(), Kokkos::AUTO()); // error
  policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
  Kokkos::parallel_for(
    "TChem::GkSurfGas::runDeviceBatch",
    policy,
    KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
      const ordinal_type i = member.league_rank();
      const real_type_1d_view_type state_at_i =
        Kokkos::subview(state, i, Kokkos::ALL());

      const real_type_1d_view_type gk_at_i = Kokkos::subview(gk, i, Kokkos::ALL());
      const real_type_1d_view_type gkSurf_at_i =
        Kokkos::subview(gkSurf, i, Kokkos::ALL());

      const real_type_1d_view_type hks_at_i = Kokkos::subview(hks, i, Kokkos::ALL());
      const real_type_1d_view_type hksSurf_at_i =
        Kokkos::subview(hksSurf, i, Kokkos::ALL());

      Scratch<real_type_1d_view_type> work(member.team_scratch(level),
                                      per_team_extent);

      auto w = (real_type*)work.data();
      auto cpks = real_type_1d_view_type(w, kmcd.nSpec);
      w += kmcd.nSpec;
      auto cpksSurf = real_type_1d_view_type(w, kmcdSurf.nSpec);
      w += kmcdSurf.nSpec;

      const Impl::StateVector<real_type_1d_view_type> sv_at_i(kmcd.nSpec,
                                                         state_at_i);
      TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                        "Error: input state vector is not valid");
      {
        const real_type t = sv_at_i.Temperature();
        const real_type p = sv_at_i.Pressure();
        const real_type_1d_view_type Xc = sv_at_i.MassFractions();

        // gas
        GkSurfGas ::team_invoke(member,
                                      t, /// input
                                      gk_at_i,
                                      hks_at_i, /// output
                                      cpks,     /// workspace
                                      kmcd);

        // surfaces
        GkSurfGas ::team_invoke(member,
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
