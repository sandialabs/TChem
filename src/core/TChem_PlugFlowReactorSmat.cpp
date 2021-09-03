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
#include "TChem_PlugFlowReactorSmat.hpp"
#include "TChem_Util.hpp"

namespace TChem {

//

template<typename PolicyType,
         typename DeviceType>
  static void
  PlugFlowReactorSmat_TemplateRun( /// input
    const std::string& profile_name,
    /// team size setting
    const PolicyType& policy,
    const Tines::value_type_2d_view<real_type, DeviceType>& state,
    const Tines::value_type_2d_view<real_type, DeviceType>& site_fraction,
    const Tines::value_type_1d_view<real_type, DeviceType>& velocity,
    const Tines::value_type_3d_view<real_type, DeviceType>& Smat,
    const Tines::value_type_3d_view<real_type, DeviceType>& Ssmat,
    const KineticModelConstData<DeviceType >& kmcd,
    const KineticSurfModelConstData<DeviceType>& kmcdSurf,
    const PlugFlowReactorData& pfrd)
    {

      using policy_type = PolicyType;
      using device_type = DeviceType;
      using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
      using real_type_2d_view_type = Tines::value_type_2d_view<real_type, device_type>;

      Kokkos::Profiling::pushRegion(profile_name);
      using policy_type = PolicyType;

      const ordinal_type level = 1;
      const ordinal_type per_team_extent = TChem::
      PlugFlowReactorSmat::getWorkSpaceSize(kmcd, kmcdSurf);

      Kokkos::parallel_for(
        profile_name,
        policy,
        KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
          const ordinal_type i = member.league_rank();
          const real_type_1d_view_type state_at_i =
            Kokkos::subview(state, i, Kokkos::ALL());
          // const real_type_0d_view velocity_at_i = Kokkos::subview(velocity, i);

          const real_type_2d_view_type Smat_at_i =
            Kokkos::subview(Smat, i, Kokkos::ALL(), Kokkos::ALL());
          const real_type_2d_view_type Ssmat_at_i =
            Kokkos::subview(Ssmat, i, Kokkos::ALL(), Kokkos::ALL());

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
            const real_type_1d_view_type Ys = sv_at_i.MassFractions();
            const real_type vel = velocity(i);
            // site fraction
            const real_type_1d_view_type Zs = Kokkos::subview(site_fraction, i, Kokkos::ALL());
            Impl::PlugFlowReactorSmat<real_type, device_type> ::team_invoke(member,
                                                    t,
                                                    Ys,
                                                    Zs,
                                                    density,
                                                    p,
                                                    vel, // input
                                                    Smat_at_i,
                                                    Ssmat_at_i, // output
                                                    work,
                                                    kmcd,
                                                    kmcdSurf,
                                                    pfrd);
          }
        });
      Kokkos::Profiling::popRegion();
    }

void
PlugFlowReactorSmat::runDeviceBatch( /// input
  const ordinal_type nBatch,
  const real_type_2d_view_type& state,
  /// input
  const real_type_2d_view_type& site_fraction,
  const real_type_1d_view_type& velocity,

  /// output
  const real_type_3d_view_type& Smat,
  const real_type_3d_view_type& Ssmat,
  /// const data from kinetic model
  const kinetic_model_type& kmcd,
  /// const data from kinetic model surface
  const kinetic_surf_model_type& kmcdSurf,
  const PlugFlowReactorData& pfrd)
{

  using policy_type = Kokkos::TeamPolicy<exec_space>;

  const ordinal_type level = 1;
  const ordinal_type per_team_extent = getWorkSpaceSize(kmcd, kmcdSurf);

  const ordinal_type per_team_scratch =
    Scratch<real_type_1d_view_type>::shmem_size(per_team_extent);

  // policy_type policy(nBatch); // error
  policy_type policy(nBatch, Kokkos::AUTO()); // fine
  // policy_type policy(nBatch, Kokkos::AUTO(), Kokkos::AUTO()); // error
  policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

  PlugFlowReactorSmat_TemplateRun( /// input
    "TChem::PlugFlowReactorSmat::runDeviceBatch",
    /// team size setting
    policy,
    state,
    site_fraction,
    velocity,
    Smat,
    Ssmat,
    kmcd,
    kmcdSurf,
    pfrd);

}

void
PlugFlowReactorSmat::runDeviceBatch( /// input
  typename UseThisTeamPolicy<exec_space>::type& policy,
  const real_type_2d_view_type& state,
  /// input
  const real_type_2d_view_type& site_fraction,
  const real_type_1d_view_type& velocity,

  /// output
  const real_type_3d_view_type& Smat,
  const real_type_3d_view_type& Ssmat,
  /// const data from kinetic model
  const kinetic_model_type& kmcd,
  /// const data from kinetic model surface
  const kinetic_surf_model_type& kmcdSurf,
  const PlugFlowReactorData& pfrd)
{

  PlugFlowReactorSmat_TemplateRun( /// input
    "TChem::PlugFlowReactorSmat::runDeviceBatch",
    /// team size setting
    policy,
    state,
    site_fraction,
    velocity,
    Smat,
    Ssmat,
    kmcd,
    kmcdSurf,
    pfrd);

}

void
PlugFlowReactorSmat::runHostBatch( /// input
  typename UseThisTeamPolicy<host_exec_space>::type& policy,
  const real_type_2d_view_host_type& state,
  /// input
  const real_type_2d_view_host_type& site_fraction,
  const real_type_1d_view_host_type& velocity,

  /// output
  const real_type_3d_view_host_type& Smat,
  const real_type_3d_view_host_type& Ssmat,
  /// const data from kinetic model
  const kinetic_model_host_type& kmcd,
  /// const data from kinetic model surface
  const kinetic_surf_model_host_type& kmcdSurf,
  const PlugFlowReactorData& pfrd)
{

  PlugFlowReactorSmat_TemplateRun( /// input
    "TChem::PlugFlowReactorSmat::runDeviceBatch",
    /// team size setting
    policy,
    state,
    site_fraction,
    velocity,
    Smat,
    Ssmat,
    kmcd,
    kmcdSurf,
    pfrd);

}

} // namespace TChem
