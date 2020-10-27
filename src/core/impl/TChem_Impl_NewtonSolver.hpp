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
#ifndef __TCHEM_IMPL_NEWTON_SOLVER_HPP__
#define __TCHEM_IMPL_NEWTON_SOLVER_HPP__

#include "TChem_Impl_DenseNanInf.hpp"
#include "TChem_Impl_DenseUTV.hpp"
#include "TChem_Util.hpp"

namespace TChem {
namespace Impl {

struct NewtonSolver
{
  template<typename ProblemType>
  KOKKOS_INLINE_FUNCTION static ordinal_type getWorkSpaceSize(
    const ProblemType& problem)
  {
    const ordinal_type m = problem.getNumberOfEquations(), n = m;
    const ordinal_type r_val = m * m + n * n + n + (m < n ? m : n) + n + n;
    /// UTV workspace for single right hand side
    return r_val;
  }

  template<typename MemberType,
           typename ProblemType,
           typename RealType1DViewType,
           typename RealType2DViewType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// intput
    const ProblemType& problem,
    const real_type& atol,
    const real_type& rtol,
    const ordinal_type& max_iter,
    /// input/output
    const RealType1DViewType& x,
    /// workspace
    const RealType1DViewType& dx,
    const RealType1DViewType& f,
    const RealType2DViewType& J,
    const RealType1DViewType& w, // workspace
                                 /// output
    /* */ ordinal_type& iter_count,
    /* */ ordinal_type& converge)
  {
    converge = false;
    real_type* wptr = w.data();
    /// the problem is square
    const ordinal_type n = problem.getNumberOfEquations();
    const ordinal_type worksize = getWorkSpaceSize(problem);
    auto work = RealType1DViewType(wptr, worksize);
    wptr += worksize;
    assert(ordinal_type(wptr - w.data()) <= ordinal_type(w.extent(0)) &&
           "Error: given workspace is smaller than required");
    problem.computeInitValues(member, x);
    bool is_valid(true);
    ordinal_type iter = 0;
    real_type norm2_f0(0);
    for (; iter < max_iter && !converge; ++iter) {
      problem.computeJacobian(member, x, J);
      problem.computeFunction(member, x, f);
      /// sanity check
      TChem::Impl::DenseNanInf ::team_check_sanity(member, J, is_valid);

      if (is_valid) {
        /// solve the equation: dx = -J^{-1} f(x);
        ordinal_type matrix_rank(0);
        TChem::Impl::DenseUTV ::team_factorize_and_solve(
          member, J, dx, f, work, matrix_rank);

#if defined(TCHEM_ENABLE_NEWTONSOLVER_USE_WRMS_NORMS)
        const real_type one(1);
        /// update the solution x and compute norm
        real_type sum(0);
        Kokkos::parallel_reduce(
          Kokkos::TeamVectorRange(member, n),
          [&](const ordinal_type& i, real_type& val) {
            x(i) -= dx(i);
            const real_type w_at_i =
              one / (rtol * ats<real_type>::abs(x(i)) + atol);
            const real_type mult_val = ats<real_type>::abs(f(i))*w_at_i;
            val += mult_val * mult_val;
          },
          sum);

        /// update norm f
        const real_type norm2_fn = ats<real_type>::sqrt(sum) / real_type(n);

        /// check convergence
        converge = norm2_fn < one;
#else
        /// update the solution x and compute norm
        real_type sum(0);
        Kokkos::parallel_reduce(
          Kokkos::TeamVectorRange(member, n),
          [&](const ordinal_type& i, real_type& val) {
            x(i) -= dx(i);
            val += ats<real_type>::abs(f(i)) * ats<real_type>::abs(f(i));
          },
          sum);

        /// update norm f
        const real_type norm2_fn = ats<real_type>::sqrt(sum) / real_type(n);
        if (iter == 0) {
          norm2_f0 = norm2_fn;
        }

        /// || f_n || < atol
        const bool a_conv = norm2_fn < atol;

        /// || f_n || / || f_0 || < rtol
        const bool r_conv = norm2_fn / norm2_f0 < rtol;

        // step tolerence is not used but it can be implemented as follows
        /// || dx_n || / || dx_{n-1} || < stol

        /// check convergence
        converge = a_conv || r_conv;
#endif
      } else {
        converge = false;
      }
    }
    /// record the final number of iterations
    iter_count = iter;
  }
};

} // namespace Impl
} // namespace TChem

#endif
