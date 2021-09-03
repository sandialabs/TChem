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
#include "TChem_PlugFlowReactorNumJacobian.hpp"

namespace TChem {

namespace Impl {

  template<typename PolicyType,
           typename DeviceType>
void
PlugFlowReactorNumJacobian_TemplateRun( /// required template arguments
  const std::string& profile_name,
  /// team size setting
  const PolicyType& policy,
  const Tines::value_type_2d_view<real_type, DeviceType>& state,
  const Tines::value_type_2d_view<real_type, DeviceType>& site_fraction,
  const Tines::value_type_1d_view<real_type, DeviceType>& velocity,
  // outputFile
  const Tines::value_type_3d_view<real_type, DeviceType>& jac,
  const Tines::value_type_2d_view<real_type, DeviceType>& fac,
  const KineticModelConstData<DeviceType >& kmcd,
  const KineticSurfModelConstData<DeviceType>& kmcdSurf,
  const PlugFlowReactorData& pfrd)
{
  Kokkos::Profiling::pushRegion(profile_name);
  using policy_type = PolicyType;
  using policy_type = PolicyType;
  using device_type = DeviceType;
  using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
  using real_type_2d_view_type = Tines::value_type_2d_view<real_type, device_type>;

  const ordinal_type level = 1;
  const ordinal_type per_team_extent = TChem::PlugFlowReactorNumJacobian
                                            ::getWorkSpaceSize(kmcd, kmcdSurf);

  const ordinal_type m = kmcd.nSpec + 3 + kmcdSurf.nSpec;

  Kokkos::parallel_for(
    profile_name,
    policy,
    KOKKOS_LAMBDA(const typename policy_type::member_type& member) {

      const ordinal_type i = member.league_rank();
      const real_type_1d_view_type state_at_i =
        Kokkos::subview(state, i, Kokkos::ALL());

      const real_type_2d_view_type jac_at_i =
        Kokkos::subview(jac, i, Kokkos::ALL(), Kokkos::ALL());
      //
      const real_type_1d_view_type fac_at_i =
        Kokkos::subview(fac, i, Kokkos::ALL());

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
        const real_type density = sv_at_i.Density();
        const real_type vel_at_i = velocity(i);
        // site fraction
        const real_type_1d_view_type Zs_at_i = Kokkos::subview(site_fraction, i, Kokkos::ALL());

        auto wptr = work.data();
        const real_type_1d_view_type vals(wptr, m);
        wptr += m;
        const real_type_1d_view_type ww(wptr,
                                    work.extent(0) - (wptr - work.data()));

        TChem::PlugFlowReactor::packToValues(
          member, t, Ys, density, vel_at_i, Zs_at_i, vals);
        member.team_barrier();

        Impl::PlugFlowReactorNumJacobian<real_type,device_type>::team_invoke(member,
                                                vals,
                                                jac_at_i,
                                                fac_at_i, // output
                                                ww,
                                                kmcd,
                                                kmcdSurf,
                                                pfrd);
      }
    });
  Kokkos::Profiling::popRegion();
}

} // namespace Impl

void
PlugFlowReactorNumJacobian::runDeviceBatch( /// thread block size
  typename UseThisTeamPolicy<exec_space>::type& policy,
  const real_type_2d_view_type& state,
  /// output
  const real_type_2d_view_type& site_fraction,
  const real_type_1d_view_type& velocity,
  // outputFile
  const real_type_3d_view_type& jac,
  const real_type_2d_view_type& fac,
  const kinetic_model_type& kmcd,
  const kinetic_surf_model_type& kmcdSurf,
  const PlugFlowReactorData& pfrd)
{
  Impl::PlugFlowReactorNumJacobian_TemplateRun(
    "TChem::PlugFlowReactorNumJacobian::runDeviceBatch",
                                 /// team policy
                                 policy,
                                 /// input
                                 state,
                                 site_fraction,
                                 velocity,
                                 //outputs
                                 jac,
                                 fac,
                                 //const parameters
                                 kmcd,
                                 kmcdSurf,
                                 pfrd);
}

void
PlugFlowReactorNumJacobian::runDeviceBatch( /// thread block size
  const ordinal_type& team_size,
  const ordinal_type& vector_size,
  const ordinal_type nBatch,
  const real_type_2d_view_type& state,
  /// output
  const real_type_2d_view_type& site_fraction,
  const real_type_1d_view_type& velocity,
  // outputFile
  const real_type_3d_view_type& jac,
  const real_type_2d_view_type& fac,
  const kinetic_model_type& kmcd,
  const kinetic_surf_model_type& kmcdSurf,
  const PlugFlowReactorData& pfrd)
{

  const auto exec_space_instance = TChem::exec_space();
  using policy_type =
    typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;

  const ordinal_type level = 1;
  const ordinal_type per_team_extent = getWorkSpaceSize(kmcd, kmcdSurf);

  const ordinal_type per_team_scratch =
    Scratch<real_type_1d_view>::shmem_size(per_team_extent);

  policy_type policy(nBatch, Kokkos::AUTO()); // fine
  if (team_size > 0 && vector_size > 0) {
    policy = policy_type(exec_space_instance, nBatch, team_size, vector_size);
  }

  policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));


  Impl::PlugFlowReactorNumJacobian_TemplateRun(
    "TChem::PlugFlowReactorNumJacobian::runDeviceBatch",
                                 /// team policy
                                 policy,
                                 /// input
                                 state,
                                 site_fraction,
                                 velocity,
                                 //outputs
                                 jac,
                                 fac,
                                 //const parameters
                                 kmcd,
                                 kmcdSurf,
                                 pfrd);
}


} // namespace TChem
