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
#include "TChem_InitialCondSurface.hpp"

namespace TChem {

void
InitialCondSurface::runDeviceBatch( /// input
  const typename TChem::UseThisTeamPolicy<TChem::exec_space>::type& policy,
  const real_type_2d_view& state,
  const real_type_2d_view& zSurf,
  /// output
  const real_type_2d_view& Z_out,
  const real_type_2d_view& fac,
  /// const data from kinetic model
  const KineticModelConstDataDevice& kmcd,
  const KineticSurfModelConstDataDevice& kmcdSurf)
{
  Kokkos::Profiling::pushRegion("TChem::InitialCondSurface::runDeviceBatch");
  using policy_type =
    typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;

  const ordinal_type level = 1;
  const ordinal_type per_team_extent = getWorkSpaceSize(kmcd, kmcdSurf);

  Kokkos::parallel_for(
    "TChem::InitialCondSurface::runDeviceBatch",
    policy,
    KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
      const ordinal_type i = member.league_rank();
      const real_type_1d_view fac_at_i =
        Kokkos::subview(fac, i, Kokkos::ALL());
      const real_type_1d_view state_at_i =
        Kokkos::subview(state, i, Kokkos::ALL());
      // site fraction
      const real_type_1d_view Zs_at_i =
        Kokkos::subview(zSurf, i, Kokkos::ALL());

      const real_type_1d_view Zs_out_at_i =
        Kokkos::subview(Z_out, i, Kokkos::ALL());

      Scratch<real_type_1d_view> work(member.team_scratch(level),
                                      per_team_extent);

      Impl::StateVector<real_type_1d_view> sv_at_i(kmcd.nSpec, state_at_i);
      TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                        "Error: input state vector is not valid");
      {
        const real_type temperature = sv_at_i.Temperature();
        const real_type pressure = sv_at_i.Pressure();
        const real_type_1d_view Ys = sv_at_i.MassFractions();

        Impl::InitialCondSurface ::team_invoke(
          member,
          temperature, /// temperature
          Ys,          /// mass fraction (kmcd.nSpec)
          pressure,    /// pressure
          Zs_at_i,
          Zs_out_at_i,
          fac_at_i,
          work, // work
          kmcd,
          kmcdSurf);

      }
    });
  Kokkos::Profiling::popRegion();
}

void
InitialCondSurface::runDeviceBatch( /// input
  const typename TChem::UseThisTeamPolicy<TChem::exec_space>::type& policy,
  const real_type_2d_view& state,
  const real_type_2d_view& zSurf,
  /// output
  const real_type_2d_view& Z_out,
  const real_type_2d_view& fac,
  /// const data from kinetic model
  const Kokkos::View<KineticModelConstDataDevice*,exec_space>& kmcds,
  const Kokkos::View<KineticSurfModelConstDataDevice*,exec_space>& kmcdSurfs)
{
  Kokkos::Profiling::pushRegion("TChem::InitialCondSurface::runDeviceBatch");
  using policy_type =
    typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;

  const ordinal_type level = 1;
  const ordinal_type per_team_extent = getWorkSpaceSize(kmcds(0), kmcdSurfs(0));

  Kokkos::parallel_for(
    "TChem::InitialCondSurface::runDeviceBatch",
    policy,
    KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
      const ordinal_type i = member.league_rank();
      const auto kmcd_at_i = (kmcds.extent(0) == 1 ? kmcds(0) : kmcds(i));
      const auto kmcd_surf_at_i = (kmcdSurfs.extent(0) == 1 ? kmcdSurfs(0) : kmcdSurfs(i));
      const real_type_1d_view fac_at_i =
        Kokkos::subview(fac, i, Kokkos::ALL());
      const real_type_1d_view state_at_i =
        Kokkos::subview(state, i, Kokkos::ALL());
      // site fraction
      const real_type_1d_view Zs_at_i =
        Kokkos::subview(zSurf, i, Kokkos::ALL());

      const real_type_1d_view Zs_out_at_i =
        Kokkos::subview(Z_out, i, Kokkos::ALL());

      Scratch<real_type_1d_view> work(member.team_scratch(level),
                                      per_team_extent);

      Impl::StateVector<real_type_1d_view> sv_at_i(kmcd_at_i.nSpec, state_at_i);
      TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                        "Error: input state vector is not valid");
      {
        const real_type temperature = sv_at_i.Temperature();
        const real_type pressure = sv_at_i.Pressure();
        const real_type_1d_view Ys = sv_at_i.MassFractions();

        Impl::InitialCondSurface ::team_invoke(
          member,
          temperature, /// temperature
          Ys,          /// mass fraction (kmcd.nSpec)
          pressure,    /// pressure
          Zs_at_i,
          Zs_out_at_i,
          fac_at_i,
          work, // work
          kmcd_at_i,
          kmcd_surf_at_i);

      }
    });
  Kokkos::Profiling::popRegion();
}

} // namespace TChem
