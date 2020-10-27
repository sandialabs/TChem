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

#include "TChem_Impl_NewtonSolver.hpp"
#include "TChem_Impl_SimpleSurface.hpp"
#include "TChem_Impl_SimpleSurface_Problem.hpp"
#include "TChem_Util.hpp"

namespace TChem {
namespace Impl {

struct InitialCondSurface
{

  template<typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType>
  KOKKOS_INLINE_FUNCTION static ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf)
  {

    using problem_type =
      TChem::Impl::SimpleSurface_Problem<KineticModelConstDataType,
                                         KineticSurfModelConstDataType>;
    problem_type problem;
    problem._kmcd = kmcd;
    problem._kmcdSurf = kmcdSurf;

    // / problem workspace
    const ordinal_type problem_workspace_size =
      problem_type::getWorkSpaceSize(kmcd, kmcdSurf);
    const ordinal_type newton_workspace_size =
      NewtonSolver::getWorkSpaceSize(problem);

    return (problem_workspace_size + newton_workspace_size +
            2 * kmcdSurf.nSpec + kmcdSurf.nSpec * kmcdSurf.nSpec);
  }

  template<typename MemberType,
           typename WorkViewType,
           typename RealType1DViewType,
           typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const real_type& t,
    const RealType1DViewType& Ys, /// (kmcd.nSpec) mass fraction
    const real_type& pressure,
    const RealType1DViewType& Zs,
    /// output
    const RealType1DViewType& Zsout,
    const RealType1DViewType& fac, /// numerica jacobian percentage
    // work
    const WorkViewType& work,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf)
  {
    using kmcd_type = KineticModelConstDataType;
    using real_type_1d_view_type = typename kmcd_type::real_type_1d_view_type;
    using real_type_2d_view_type = typename kmcd_type::real_type_2d_view_type;

    // do newton solver
    const real_type atol_newton = 1e-12, rtol_newton = 1e-8;
    const ordinal_type max_num_newton_iterations(1000);

    using problem_type =
      TChem::Impl::SimpleSurface_Problem<KineticModelConstDataType,
                                         KineticSurfModelConstDataType>;
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
    problem._x = Zs;              // initial conditions
    problem._fac = fac;    // fac for numerical jacobian

    const ordinal_type m = kmcdSurf.nSpec;
    /// newton workspace

    auto dx = real_type_1d_view_type(wptr, m);
    wptr += m;
    auto f = real_type_1d_view_type(wptr, m);
    wptr += m;
    auto J = real_type_2d_view_type(wptr, m, m);
    wptr += m * m;
    const ordinal_type newton_work_size =
      NewtonSolver::getWorkSpaceSize(problem);
    auto w = real_type_1d_view_type(wptr, newton_work_size);
    wptr += newton_work_size;

    /// error check
    const ordinal_type workspace_used(wptr - work.data()),
      workspace_extent(work.extent(0));
    if (workspace_used > workspace_extent) {
      printf("workspace_used %d workspace_extent %d\n",
             workspace_used,
             workspace_extent);
      Kokkos::abort("Error: workspace used is larger than it is "
                    "provided::TChem_Impl_InitialCondSurface\n");
    }

    ordinal_type converge(0);
    ordinal_type newton_iteration_count(0);
    TChem::Impl::NewtonSolver::team_invoke(member,
                                            problem,
                                            atol_newton,
                                            rtol_newton,
                                            max_num_newton_iterations,
                                            Zsout,
                                            dx,
                                            f,
                                            J,
                                            w,
                                            newton_iteration_count,
                                            converge);
    member.team_barrier();

    if (!converge) {
      printf(" Newton solver does not coverge in InitialCondSurface \n");
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
        fprintf(fs, "   i %3d,  Zs %e, \n", i, Zs(i));

      fprintf(fs, ":::: output\n");
      for (int i = 0; i < kmcdSurf.nSpec; ++i)
        fprintf(fs, "   i %3d,  Zs %e, \n", i, Zsout(i));
    }
#endif
  }
};

} // namespace Impl
} // namespace TChem

#endif
