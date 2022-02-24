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
#ifndef __TCHEM_IMPL_ISOTHERMAL_TRANSIENT_CONT_STIRRED_TANK_REACTOR_HPP__
#define __TCHEM_IMPL_ISOTHERMAL_TRANSIENT_CONT_STIRRED_TANK_REACTOR_HPP__

#include "TChem_Util.hpp"
#include "TChem_Impl_IsothermalTransientContStirredTankReactor_Problem.hpp"


namespace TChem {
namespace Impl {

template<typename ValueType, typename DeviceType>
struct IsothermalTransientContStirredTankReactor
{

  using value_type = ValueType;
  using device_type = DeviceType;
  using scalar_type = typename ats<value_type>::scalar_type;

  using real_type = scalar_type;
  using real_type_0d_view_type = Tines::value_type_0d_view<real_type,device_type>;
  using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;
  using real_type_2d_view_type = Tines::value_type_2d_view<real_type,device_type>;

  using kinetic_model_type = KineticModelConstData<device_type>;
  using kinetic_surf_model_type = KineticSurfModelConstData<device_type>;
  using cstr_data_type = TransientContStirredTankReactorData<device_type>;

  using TimeIntegrator = Tines::TimeIntegratorTrBDF2<value_type, device_type>;

  static inline ordinal_type getWorkSpaceSize(
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf)
  {
    //
    using problem_type =
      TChem::Impl::IsothermalTransientContStirredTankReactor_Problem<value_type, device_type>;;

    problem_type problem;
    problem._kmcd = kmcd;
    problem._kmcdSurf = kmcdSurf;
    ordinal_type problem_workspace_size = problem.getWorkSpaceSize();
    ordinal_type m = problem.getNumberOfEquations();
    ordinal_type worksizeTimeIntegration(0);
    TimeIntegrator::workspace(m, worksizeTimeIntegration);

    return worksizeTimeIntegration + problem_workspace_size;
  }

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input iteration and qoi index to store
    const ordinal_type& jacobian_interval,
    const ordinal_type& max_num_newton_iterations,
    const ordinal_type& max_num_time_iterations,
    /// input time step and time range
    const real_type_1d_view_type& tol_newton,
    const real_type_2d_view_type& tol_time,
    const real_type_1d_view_type& fac, /// numerica jacobian percentage
    /// input time step and time range
    const real_type& dt_in,
    const real_type& dt_min,
    const real_type& dt_max,
    const real_type& t_beg,
    const real_type& t_end,
    /// input (initial condition)
    const real_type temperature,
    const real_type_1d_view_type& vals,
    /// output (final output conditions)
    const real_type_0d_view_type& t_out,
    const real_type_0d_view_type& dt_out,
    const real_type_1d_view_type& vals_out,
    const real_type_0d_view_type& mass_flow_out,
    /// workspace
    const real_type_1d_view_type& work,
    /// const input from kinetic model
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf,
    const cstr_data_type& cstr)
  {

    using problem_type =
      TChem::Impl::IsothermalTransientContStirredTankReactor_Problem<value_type, device_type>;
    problem_type problem;

    /// problem workspace
    const ordinal_type problem_workspace_size =
      problem_type ::getWorkSpaceSize(kmcd, kmcdSurf);

    auto wptr = work.data();
    auto pw = real_type_1d_view(wptr, problem_workspace_size);
    wptr += problem_workspace_size;

    /// error check
    const ordinal_type workspace_used(wptr - work.data()),
      workspace_extent(work.extent(0));
    if (workspace_used > workspace_extent) {
      Kokkos::abort("Error: workspace used is larger than it is provided\n");
    }
    /// time integrator workspace
    auto tw = real_type_1d_view_type(wptr, workspace_extent - workspace_used);

    /// constant values of the problem
    problem._kmcd = kmcd;         // kinetic model
    problem._kmcdSurf = kmcdSurf; // surface kinetic model
    problem._cstr = cstr;
    problem._work = pw; // problem workspace array
    problem._fac = fac; // fac for numerical jacobian
    problem._temperature = temperature;
    problem._mass_flow_out = mass_flow_out;// this is output

    TimeIntegrator::invoke(member,
                           problem,
                           jacobian_interval,
                           max_num_newton_iterations,
                           max_num_time_iterations,
                           tol_newton,
                           tol_time,
                           dt_in,
                           dt_min,
                           dt_max,
                           t_beg,
                           t_end,
                           vals,
                           t_out,
                           dt_out,
                           vals_out,
                           tw);

  }
};

} // namespace Impl
} // namespace TChem

#endif
