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
#ifndef __TCHEM_IMPL_INITIALCONDSURFACE_HPP__
#define __TCHEM_IMPL_INITIALCONDSURFACE_HPP__

#include "TChem_Impl_SimpleSurface.hpp"
#include "TChem_Impl_SimpleSurface_Problem.hpp"
#include "TChem_Impl_AlgebraicConstraintsSurface_Problem.hpp"
#include "TChem_Util.hpp"


namespace TChem {
namespace Impl {

template<typename ValueType, typename DeviceType>
struct InitialCondSurface
{
  using value_type = ValueType;
  using device_type = DeviceType;
  using scalar_type = typename ats<value_type>::scalar_type;

  using real_type = scalar_type;
  using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;
  using real_type_2d_view_type = Tines::value_type_2d_view<real_type,device_type>;

  using kinetic_model_type      = KineticModelConstData<device_type>;
  using kinetic_surf_model_type = KineticSurfModelConstData<device_type>;

  KOKKOS_INLINE_FUNCTION static ordinal_type getWorkSpaceSize(
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf)
  {

    using problem_type =
      TChem::Impl::SimpleSurface_Problem<real_type,
                                         device_type>;
    problem_type problem;
    const ordinal_type m = kmcdSurf.nSpec;
    using newton_solver_type = Tines::NewtonSolver<real_type, device_type>;
    ordinal_type wlen_newton(0);
      newton_solver_type::workspace(m, wlen_newton); /// utv workspace
    // / problem workspace
    const ordinal_type problem_workspace_size =
      problem_type::getWorkSpaceSize(kmcd, kmcdSurf);

    return (problem_workspace_size + wlen_newton + m+
            2 * kmcdSurf.nSpec + kmcdSurf.nSpec * kmcdSurf.nSpec);
  }

  template<typename MemberType,
           typename WorkViewType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// numerical inputs
    const real_type_1d_view_type& tol_newton,
    const real_type& max_num_newton_iterations,
    // condition inputs
    const real_type& t,
    const real_type_1d_view_type& Ys, /// (kmcd.nSpec) mass fraction
    const real_type& pressure,
    const real_type_1d_view_type& site_fraction,
    /// output
    const real_type_1d_view_type& site_fraction_out,
    const real_type_1d_view_type& fac, /// numerica jacobian percentage
    // work
    const WorkViewType& work,
    /// const input from kinetic model
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf)
  {

    using newton_solver_type = Tines::NewtonSolver<real_type, device_type>;

    // do newton solver
    // const real_type atol_newton = 1e-12, rtol_newton = 1e-8;
    // const ordinal_type max_num_newton_iterations(1000);

    using problem_type =
      TChem::Impl::SimpleSurface_Problem<real_type,
                                         device_type>;
    problem_type problem;

    /// problem workspace
    const ordinal_type problem_workspace_size =
      problem_type::getWorkSpaceSize(kmcd, kmcdSurf);

    auto wptr = work.data();
    auto pw = real_type_1d_view_type(wptr, problem_workspace_size);
    wptr += problem_workspace_size;

    /// constant values of the problem
    problem._p = pressure;        // pressure
    problem._kmcd = kmcd;         // kinetic model
    problem._kmcdSurf = kmcdSurf; // surface kinetic model
    problem._Ys = Ys;             // mass fraction is constant
    problem._t = t;               // temperature
    problem._work = pw;           // problem workspace array
    problem._x = site_fraction;              // initial conditions
    problem._fac = fac;    // fac for numerical jacobian

    const ordinal_type m = kmcdSurf.nSpec;
    /// newton workspace

    auto dx = real_type_1d_view_type(wptr, m);
    wptr += m;
    auto f = real_type_1d_view_type(wptr, m);
    wptr += m;
    auto J = real_type_2d_view_type(wptr, m, m);
    wptr += m * m;

    ordinal_type newton_work_size(0);
    newton_solver_type::workspace(m, newton_work_size); /// utv workspace
    auto w = real_type_1d_view_type(wptr, newton_work_size);
    wptr += newton_work_size;

    /// error check


    using algebraic_constraint_surface_problem_type =
        AlgebraicConstraintsSurface_Problem<real_type, device_type>;

    algebraic_constraint_surface_problem_type algebraic_constraints;
    algebraic_constraints._problem = problem;
    auto unr = real_type_1d_view_type(wptr, m);
    wptr += m;
    algebraic_constraints._unr = unr;


    const ordinal_type workspace_used(wptr - work.data()),
      workspace_extent(work.extent(0));
    if (workspace_used > workspace_extent) {
      // printf("workspace_used %d workspace_extent %d\n",
             // workspace_used,
             // workspace_extent);
      Kokkos::abort("Error: workspace used is larger than it is "
                    "provided::TChem_Impl_InitialCondSurface\n");
    }

    ordinal_type converge(0);
    int newton_iteration_count(0);
    newton_solver_type::invoke(member, algebraic_constraints,
                              tol_newton(0), //atol_newton
                              tol_newton(1), // rtol_newton
                              max_num_newton_iterations,
                              site_fraction_out,
                              dx,
                              f,
                              J,
                              w,
                              newton_iteration_count,
                              converge);

    member.team_barrier();

    if (!converge) {
      Kokkos::abort("Error: Initial Conditions surface");
    }

#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("InitialCondSurface.team_invoke.test.out", "a+");
      fprintf(fs, ":: InitialCondSurface::team_invoke\n");
      fprintf(fs, ":::: input\n");
      fprintf(fs,
              "     nSpec %3d, nReac %3d, site density %e\n",
              kmcdSurf.nSpec,
              kmcdSurf.nReac,
              kmcdSurf.sitedensity);
      fprintf(fs, "  t %e, p %e\n", t, pressure);
      for (int i = 0; i < kmcd.nSpec; ++i)
        fprintf(fs, "   i %3d,  Ys %e, \n", i, Ys(i));
      for (int i = 0; i < kmcdSurf.nSpec; ++i)
        fprintf(fs, "   i %3d,  site_fraction %e, \n", i, site_fraction(i));

      fprintf(fs, ":::: output\n");
      for (int i = 0; i < kmcdSurf.nSpec; ++i)
        fprintf(fs, "   i %3d,  site_fraction %e, \n", i, site_fraction_out(i));
    }
#endif
  }
};

} // namespace Impl
} // namespace TChem

#endif
