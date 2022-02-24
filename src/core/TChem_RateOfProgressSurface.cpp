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

#include "TChem_Impl_RateOfProgressSurfaceInd.hpp"

#include "TChem_RateOfProgressSurface.hpp"

namespace TChem {
  template<typename PolicyType,
           typename DeviceType>
  void
  RateOfProgressSurface_TemplateRun( /// input
    const std::string& profile_name,
    /// team size setting
    const PolicyType& policy,
    const Tines::value_type_2d_view<real_type, DeviceType>& state,
    const Tines::value_type_2d_view<real_type, DeviceType>& site_fraction,
    /// output
    const Tines::value_type_2d_view<real_type, DeviceType>& RoPFor,
    const Tines::value_type_2d_view<real_type, DeviceType>& RoPRev,
    /// const data from kinetic model
    const KineticModelConstData<DeviceType >& kmcd,
    /// const data from kinetic model surface
    const KineticSurfModelConstData<DeviceType>& kmcdSurf)
  {
    Kokkos::Profiling::pushRegion(profile_name);
    using policy_type = PolicyType;
    using device_type = DeviceType;

    const ordinal_type level = 1;
    using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;

    const ordinal_type per_team_extent = RateOfProgressSurface::getWorkSpaceSize(kmcd, kmcdSurf);

    Kokkos::parallel_for(
      profile_name,
      policy,
      KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
        const ordinal_type i = member.league_rank();
        const real_type_1d_view_type state_at_i =
          Kokkos::subview(state, i, Kokkos::ALL());
        const real_type_1d_view_type Zk_at_i =
          Kokkos::subview(site_fraction, i, Kokkos::ALL());
        const real_type_1d_view_type RoPFor_at_i =
          Kokkos::subview(RoPFor, i, Kokkos::ALL());
        const real_type_1d_view_type RoPRev_at_i =
          Kokkos::subview(RoPRev, i, Kokkos::ALL());

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
          // when density is lower than zero, tchem will compute it. 
          const real_type density = sv_at_i.Density() > 0 ? sv_at_i.Density() :
                            Impl::RhoMixMs<real_type,device_type>
                                ::team_invoke(member,  sv_at_i.Temperature(),
                                sv_at_i.Pressure(),
                                sv_at_i.MassFractions(), kmcd);
          member.team_barrier();

          Impl::RateOfProgressSurfaceInd<real_type, device_type> ::team_invoke(member,
                                                       t,
                                                       p,
                                                       density,
                                                       Ys,
                                                       Zk_at_i,
                                                       RoPFor_at_i,
                                                       RoPRev_at_i,
                                                       work,
                                                       kmcd,
                                                       kmcdSurf);
        }
      });
    Kokkos::Profiling::popRegion();
  }


void
RateOfProgressSurface::runDeviceBatch( /// input
  const ordinal_type nBatch,
  const real_type_2d_view_type& state,
  const real_type_2d_view_type& site_fraction,
  /// output
  const real_type_2d_view_type& RoPFor,
  const real_type_2d_view_type& RoPRev,
  /// const data from kinetic model
  const kinetic_model_type& kmcd,
  /// const data from kinetic model surface
  const kinetic_surf_model_type& kmcdSurf)
{
  using policy_type = Kokkos::TeamPolicy<exec_space>;

  const ordinal_type level = 1;
  const ordinal_type per_team_extent = getWorkSpaceSize(kmcd, kmcdSurf);
  const ordinal_type per_team_scratch =
    Scratch<real_type_1d_view>::shmem_size(per_team_extent);

  // policy_type policy(nBatch); // error
  policy_type policy(nBatch, Kokkos::AUTO()); // fine
  // policy_type policy(nBatch, Kokkos::AUTO(), Kokkos::AUTO()); // error
  policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

  RateOfProgressSurface_TemplateRun( /// input
    "TChem::RateOfProgressSurface::runDeviceBatch",
    /// team size setting
    policy,
    state,
    site_fraction,
    /// output
    RoPFor,
    RoPRev,
    /// const data from kinetic model
    kmcd,
    /// const data from kinetic model surface
    kmcdSurf);

}

void
RateOfProgressSurface::runDeviceBatch( /// input
  typename UseThisTeamPolicy<exec_space>::type& policy,
  const real_type_2d_view_type& state,
  const real_type_2d_view_type& site_fraction,
  /// output
  const real_type_2d_view_type& RoPFor,
  const real_type_2d_view_type& RoPRev,
  /// const data from kinetic model
  const kinetic_model_type& kmcd,
  /// const data from kinetic model surface
  const kinetic_surf_model_type& kmcdSurf)
{

  RateOfProgressSurface_TemplateRun( /// input
    "TChem::RateOfProgressSurface::runDeviceBatch",
    /// team size setting
    policy,
    state,
    site_fraction,
    /// output
    RoPFor,
    RoPRev,
    /// const data from kinetic model
    kmcd,
    /// const data from kinetic model surface
    kmcdSurf);

}

} // namespace TChem
