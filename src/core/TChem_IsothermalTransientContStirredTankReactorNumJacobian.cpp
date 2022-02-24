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

#include "TChem_IsothermalTransientContStirredTankReactorNumJacobian.hpp"

namespace TChem {


  template<typename PolicyType,
           typename DeviceType>
  void
  IsothermalTransientContStirredTankReactorNumJacobian_TemplateRun( /// required template arguments
    const std::string& profile_name,
    const PolicyType& policy,
    // inputs
    const Tines::value_type_2d_view<real_type, DeviceType> & state,
    const Tines::value_type_2d_view<real_type, DeviceType> & site_fraction,
    /// output
    /// const data from kinetic model
    const Tines::value_type_3d_view<real_type, DeviceType> & jac,
    const KineticModelConstData<DeviceType >& kmcd,
    const KineticSurfModelConstData<DeviceType>& kmcdSurf,
    const TransientContStirredTankReactorData<DeviceType>& cstr)
  {

    Kokkos::Profiling::pushRegion(profile_name);
    using policy_type = PolicyType;
    using device_type = DeviceType;
    using real_type_0d_view_type = Tines::value_type_0d_view<real_type, device_type>;
    using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
    using real_type_2d_view_type = Tines::value_type_2d_view<real_type, device_type>;

    using problem_type =
      TChem::Impl::IsothermalTransientContStirredTankReactor_Problem<real_type,device_type>;

    const ordinal_type level = 1;

    const ordinal_type per_team_extent =
     TChem::IsothermalTransientContStirredTankReactorNumJacobian
          ::getWorkSpaceSize(kmcd, kmcdSurf); ///

    const ordinal_type per_team_scratch =
        Scratch<real_type_1d_view_type>::shmem_size(per_team_extent);

    const ordinal_type m = problem_type::getNumberOfEquations(kmcd, kmcdSurf);

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
          const real_type_1d_view_type fac(wptr, m);
          wptr += m;
          const real_type_1d_view_type mass_flow_out_work(wptr,1);
          wptr += 1;
          const real_type_0d_view_type mass_flow_out = Kokkos::subview(mass_flow_out_work, 0);;
          const real_type_1d_view_type ww(wptr,
                                      work.extent(0) - (wptr - work.data()));

          TChem::IsothermalTransientContStirredTankReactor::packToValues(
            member, Ys, site_fraction_at_i, vals);

          member.team_barrier();

          problem_type problem;
          /// constant values of the problem
          problem._kmcd = kmcd;         // kinetic model
          problem._kmcdSurf = kmcdSurf; // surface kinetic model
          problem._cstr = cstr;
          problem._work = ww; // problem workspace array
          problem._fac = fac;    // fac for numerical jacobian
          problem._temperature = temperature;
          problem._mass_flow_out = mass_flow_out;// this is output

          problem.computeNumericalJacobianRichardsonExtrapolation(member, vals, jac_at_i);
          member.team_barrier();

        }
      });
    Kokkos::Profiling::popRegion();
  }

  void
  IsothermalTransientContStirredTankReactorNumJacobian::runDeviceBatch( /// thread block size
    typename UseThisTeamPolicy<exec_space>::type& policy,
    const real_type_2d_view_type& state,
    const real_type_2d_view_type& site_fraction,
    /// output
    const real_type_3d_view_type& jac,
    /// const data from kinetic model
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf,
    const cstr_data_type& cstr)
  {
    IsothermalTransientContStirredTankReactorNumJacobian_TemplateRun( /// template arguments deduction
      "TChem::IsothermalTransientContStirredTankReactorNumJacobian::runDeviceBatch",
      policy,
      state,
      site_fraction,
      jac,
      /// const data of kinetic model
      kmcd,
      kmcdSurf,
      cstr);
  }

} // namespace TChem
