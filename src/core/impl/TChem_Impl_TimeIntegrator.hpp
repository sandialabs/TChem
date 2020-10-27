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
#ifndef __TCHEM_IMPL_TIME_INTEGRATOR_HPP__
#define __TCHEM_IMPL_TIME_INTEGRATOR_HPP__

#include "TChem_Util.hpp"

#include "TChem_Impl_NewtonSolver.hpp"
#include "TChem_Impl_TrBDF2.hpp"

namespace TChem {
namespace Impl {

struct TimeIntegrator
{
  template<typename ProblemType>
  KOKKOS_INLINE_FUNCTION static ordinal_type getWorkSpaceSize(
    const ProblemType& problem)
  {
    using problem_type = ProblemType;

    const ordinal_type problem_workspace_size = problem.getWorkSpaceSize();
    const ordinal_type trbdf_workspace_size =
      TrBDF2<typename problem_type::exec_space_type>::getWorkSpaceSize(problem);
    const ordinal_type newton_workspace_size =
      NewtonSolver::getWorkSpaceSize(problem);

    return (problem_workspace_size + trbdf_workspace_size +
            newton_workspace_size);
  }

  template<typename MemberType,
           typename ProblemType,
           typename WorkViewType,
           typename RealType0DViewType,
           typename RealType1DViewType,
           typename RealType2DViewType>
  KOKKOS_INLINE_FUNCTION static ordinal_type team_invoke_detail(
    const MemberType& member,
    /// problem
    const ProblemType& problem,
    /// input iteration and qoi index to store
    const ordinal_type& max_num_newton_iterations,
    const ordinal_type& max_num_time_iterations,
    const RealType1DViewType& tol_newton,
    const RealType2DViewType& tol_time,
    /// input time step and time range
    const real_type& dt_in,
    const real_type& dt_min,
    const real_type& dt_max,
    const real_type& t_beg,
    const real_type& t_end,
    /// input (initial condition)
    const RealType1DViewType& vals,
    /// output (final output conditions)
    const RealType0DViewType& t_out,
    const RealType0DViewType& dt_out,
    const RealType1DViewType& vals_out,
    /// workspace
    const WorkViewType& work)
  {
    using problem_type = ProblemType;
    using real_type_1d_view_type =
      typename problem_type::real_type_1d_view_type;
    using real_type_2d_view_type =
      typename problem_type::real_type_2d_view_type;

    /// return value; when it fails it return non-zero value
    ordinal_type r_val(0);

    /// const values
    const real_type zero(0), half(1 / 2), two(2), minus_one(-1);

    /// early return
    if (dt_in < zero)
      return 3;

    /// data structure here is temperature, mass fractions of species...
    const ordinal_type m = problem.getNumberOfEquations(),
                       m_ode = problem.getNumberOfTimeODEs();
    ;

    /// time stepping object
    TChem::Impl::TrBDF2<typename problem_type::exec_space_type> trbdf;
    TChem::Impl::TrBDF2_Part1<problem_type> trbdf_part1;
    TChem::Impl::TrBDF2_Part2<problem_type> trbdf_part2;

    /// TrBDF2 parameters
    const real_type gamma = two - TChem::ats<real_type>::sqrt(two);
    trbdf._gamma = gamma;
    trbdf_part1._gamma = gamma;
    trbdf_part2._gamma = gamma;

    /// workspace
    auto wptr = work.data();

    /// trbdf workspace
    auto u = real_type_1d_view_type(wptr, m);
    wptr += m;
    auto un = real_type_1d_view_type(wptr, m);
    wptr += m;
    auto unr = real_type_1d_view_type(wptr, m);
    wptr += m;
    auto fn = real_type_1d_view_type(wptr, m);
    wptr += m;
    auto fnr = real_type_1d_view_type(wptr, m);
    wptr += m;

    /// newton workspace
    auto dx = real_type_1d_view_type(wptr, m);
    wptr += m;
    auto f = real_type_1d_view_type(wptr, m);
    wptr += m;
    auto J = real_type_2d_view_type(wptr, m, m);
    wptr += m * m;
    // auto w   = real_type_1d_view(wptr, 2*m); wptr += (2*m);
    const ordinal_type newton_workspace_size =
      NewtonSolver::getWorkSpaceSize(problem);
    auto w = real_type_1d_view_type(wptr, newton_workspace_size);
    wptr += (newton_workspace_size);

    /// error check
    const ordinal_type workspace_used(wptr - work.data()),
      workspace_extent(work.extent(0));
    if (workspace_used > workspace_extent) {
      Kokkos::abort("Error: workspace used is larger than it is provided\n");
    }

    /// assign the problem to trbdf
    trbdf_part1._problem = problem;
    trbdf_part2._problem = problem;

    /// initial conditions
    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                         [&](const ordinal_type& k) { un(k) = vals(k); });
    member.team_barrier();

    /// time integration
    real_type t(t_beg), dt(dt_in);
    for (ordinal_type iter = 0; iter < max_num_time_iterations && dt != zero;
         ++iter) {
      {
        ordinal_type converge(0);
        for (ordinal_type i = 0; i < 4 && converge == 0; ++i) {
          ordinal_type converge_part1(0);
          {
            dt = (dt > dt_min ? dt : dt_min);
            trbdf_part1._dt = dt;
            trbdf_part1._un = un;
            trbdf_part1._fn = fn;
            problem.computeFunction(member, un, fn);

            ordinal_type newton_iteration_count(0);
            TChem::Impl::NewtonSolver ::team_invoke(member,
                                                    trbdf_part1,
                                                    tol_newton(0),
                                                    tol_newton(1),
                                                    max_num_newton_iterations,
                                                    unr,
                                                    dx,
                                                    f,
                                                    J,
                                                    w,
                                                    newton_iteration_count,
                                                    converge_part1);

            if (converge_part1) {
              problem.computeFunction(member, unr, fnr);
            } else {
              /// try again with half time step
              dt *= half;
              continue;
            }
          }

          ordinal_type converge_part2(0);
          {
            trbdf_part2._dt = dt;
            trbdf_part2._un = un;
            trbdf_part2._unr = unr;

            ordinal_type newton_iteration_count(0);
            TChem::Impl::NewtonSolver ::team_invoke(member,
                                                    trbdf_part2,
                                                    tol_newton(0),
                                                    tol_newton(1),
                                                    max_num_newton_iterations,
                                                    u,
                                                    dx,
                                                    f,
                                                    J,
                                                    w,
                                                    newton_iteration_count,
                                                    converge_part2);
            if (converge_part2) {
              problem.computeFunction(member, u, f);
            } else {
              dt *= half;
              continue;
            }
          }
          converge = converge_part1 && converge_part2;
        }

        if (converge) {
          t += dt;
          trbdf.computeTimeStepSize(
            member, dt_min, dt_max, tol_time, m_ode, fn, fnr, f, u, dt);
          dt = ((t + dt) > t_end) ? t_end - t : dt;
          Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                               [&](const ordinal_type& k) { un(k) = u(k); });
        } else {
          Kokkos::single(Kokkos::PerTeam(member), [&]() {
            printf("Warning: TimeIntegrator, sample (%d) trbdf fails to "
                   "converge with current time step %e\n",
                   int(member.league_rank()),
                   dt);
          });
          r_val = 1;
          break;
        }
      }
      member.team_barrier();
    }

    {
      /// finalize with output for next iterations of time solutions
      if (r_val == 0) {
        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                             [&](const ordinal_type& k) {
                               vals_out(k) = u(k);
                               if (k == 0) {
                                 t_out() = t;
                                 dt_out() = dt;
                               }
                             });
      } else {
        /// if newton fails,
        /// - set values with zero
        /// - t_out becomes t_end
        /// - dt_out is minus one
        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                             [&](const ordinal_type& k) {
                               vals_out(k) = zero;
                               if (k == 0) {
                                 t_out() = t_end;
                                 dt_out() = minus_one;
                               }
                             });
      }
    }

    return r_val;
  }
};

} // namespace Impl
} // namespace TChem

#endif
