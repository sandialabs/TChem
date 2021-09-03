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

#include "TChem_TransientContStirredTankReactorNumJacobian.hpp"

namespace TChem {


  template<typename PolicyType,
           typename DeviceType>
  void
  TransientContStirredTankReactorNumJacobian_TemplateRun( /// required template arguments
    const std::string& profile_name,
    const PolicyType& policy,
    // inputs
    const Tines::value_type_2d_view<real_type, DeviceType> & state,
    const Tines::value_type_2d_view<real_type, DeviceType> & site_fraction,
    /// output
    /// const data from kinetic model
    const Tines::value_type_3d_view<real_type, DeviceType> & jac,
    const Tines::value_type_2d_view<real_type, DeviceType> & fac,
    const KineticModelConstData<DeviceType >& kmcd,
    const KineticSurfModelConstData<DeviceType>& kmcdSurf,
    const TransientContStirredTankReactorData<DeviceType>& cstr)
  {

    Kokkos::Profiling::pushRegion(profile_name);
    using policy_type = PolicyType;
    using device_type = DeviceType;
    using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
    using real_type_2d_view_type = Tines::value_type_2d_view<real_type, device_type>;

    const ordinal_type level = 1;

    const ordinal_type per_team_extent =
     TChem::TransientContStirredTankReactorNumJacobian
          ::getWorkSpaceSize(kmcd, kmcdSurf); ///

    const ordinal_type per_team_scratch =
        Scratch<real_type_1d_view_type>::shmem_size(per_team_extent);

    const ordinal_type m = Impl::TransientContStirredTankReactor_Problem
                  <real_type,device_type>::getNumberOfEquations(kmcd, kmcdSurf);

    Kokkos::parallel_for(
      profile_name,
      policy,
      KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
        const ordinal_type i = member.league_rank();
        const real_type_1d_view_type state_at_i =
          Kokkos::subview(state, i, Kokkos::ALL());
        // site fraction
        const real_type_1d_view_type site_fraction_at_i =
          Kokkos::subview(site_fraction, i, Kokkos::ALL());

        const real_type_2d_view_type jac_at_i =
        Kokkos::subview(jac, i, Kokkos::ALL(), Kokkos::ALL());

        const real_type_1d_view_type fac_at_i =
          Kokkos::subview(fac, i, Kokkos::ALL());

        Scratch<real_type_1d_view_type> work(member.team_scratch(level),
                                         per_team_extent);

        Impl::StateVector<real_type_1d_view_type> sv_at_i(kmcd.nSpec, state_at_i);
        TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                          "Error: input state vector is not valid");
        {

          const real_type temperature = sv_at_i.Temperature();
          const real_type pressure = sv_at_i.Pressure();
          const real_type density = sv_at_i.Density();
          const real_type_1d_view_type Ys = sv_at_i.MassFractions();

          auto wptr = work.data();
          const real_type_1d_view_type vals(wptr, m);
          wptr += m;
          const real_type_1d_view_type ww(wptr,
                                      work.extent(0) - (wptr - work.data()));

          //
          TChem::TransientContStirredTankReactor::packToValues(
            member, temperature,  Ys, site_fraction_at_i, vals);

          member.team_barrier();

          Impl::TransientContStirredTankReactorNumJacobian<real_type,device_type> ::team_invoke(member,
                                                  vals,
                                                  jac_at_i,
                                                  fac_at_i, // output
                                                  ww,
                                                  kmcd,
                                                  kmcdSurf,
                                                  cstr);



        }
      });
    Kokkos::Profiling::popRegion();
  }

  void
  TransientContStirredTankReactorNumJacobian::runDeviceBatch( /// thread block size
    const ordinal_type nBatch,
    const real_type_2d_view_type& state,
    const real_type_2d_view_type& site_fraction,
    /// output
    const real_type_3d_view_type& jac,
    const real_type_2d_view_type& fac,
    /// const data from kinetic model
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf,
    const cstr_data_type& cstr)
  {
    // setting policy
    const auto exec_space_instance = TChem::exec_space();
    using policy_type =
      typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;
    const ordinal_type level = 1;
    const ordinal_type per_team_extent =
     TChem::TransientContStirredTankReactorNumJacobian
          ::getWorkSpaceSize(kmcd, kmcdSurf); ///
    const ordinal_type per_team_scratch =
        Scratch<real_type_1d_view_type>::shmem_size(per_team_extent);
    policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());

    policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

    TransientContStirredTankReactorNumJacobian_TemplateRun( /// template arguments deduction
      "TChem::TransientContStirredTankReactorNumJac::runDeviceBatch",
      policy,
      state,
      site_fraction,
      jac,
      fac,
      /// const data of kinetic model
      kmcd,
      kmcdSurf,
      cstr);
  }

} // namespace TChem
