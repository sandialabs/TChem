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
#ifndef __TCHEM_IMPL_TR_BDF2_HPP__
#define __TCHEM_IMPL_TR_BDF2_HPP__

#include "TChem_Util.hpp"

namespace TChem {
namespace Impl {

template<typename ExecSpace>
struct TrBDF2
{
  static constexpr bool is_device = std::is_same<ExecSpace, exec_space>::value;
  using real_type_1d_view_type = typename std::
    conditional<is_device, real_type_1d_view, real_type_1d_view_host>::type;
  using real_type_2d_view_type = typename std::
    conditional<is_device, real_type_2d_view, real_type_2d_view_host>::type;

  KOKKOS_DEFAULTED_FUNCTION
  TrBDF2() = default;

  real_type _gamma;

#if defined(TCHEM_ENABLE_TRBDF2_USE_WRMS_NORMS)
  /// IDA WRMS version
  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION void computeTimeStepSize(
    const MemberType& member,
    const real_type& dtmin,
    const real_type& dtmax,
    const real_type_2d_view_type& tol,
    const ordinal_type& m, // vector length
    const real_type_1d_view_type& fn,
    const real_type_1d_view_type& fnr,
    const real_type_1d_view_type& fnp,
    const real_type_1d_view_type& u,
    /* */ real_type& dt)
  {
    const real_type two(2), kr = (-3.0 * _gamma * _gamma + 4.0 * _gamma - 2.0) /
                                 (12.0 * (2.0 - _gamma));
    const real_type one(1), scal1(one / _gamma), scal2(one / (one - _gamma));
    const real_type half(0.5);
    using reducer_value_type = typename Kokkos::Sum<real_type>::value_type;
    reducer_value_type norm;
    Kokkos::Sum<real_type> reducer_value(norm);
    Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(member, m),
      [&](const ordinal_type& i, reducer_value_type& update) {
        /// error estimation
        const real_type abs_est_err = ats<real_type>::abs(
          two * kr * dt *
          (scal1 * fn(i) - scal1 * scal2 * fnr(i) + scal2 * fnp(i)));
        const real_type w_at_i =
          one / (tol(i, 1) * ats<real_type>::abs(u(i)) + tol(i, 0));
        const real_type mult_val = abs_est_err * w_at_i;
        update += mult_val * mult_val;
      },
      reducer_value);
    norm = (ats<real_type>::sqrt(norm / real_type(m)));
    {
      /// WRMS is close to zero, the error is reasonably small
      /// we do not know how large is large enough to reduce time step size
      /// here we just set 10
      const real_type alpha =
        norm < one ? two : norm > real_type(10) ? half : one;
      const real_type dtnew = dt * alpha;
      if (alpha < one) {
        dt = dtnew < dtmin ? dtmin : dtnew;
      } else {
        dt = dtnew > dtmax ? dtmax : dtnew;
      }
    }
  }
#else
  /// in this scheme, we do not use atol as we normalize the error
  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION void computeTimeStepSize(
    const MemberType& member,
    const real_type& dtmin,
    const real_type& dtmax,
    const real_type_2d_view_type& tol,
    const ordinal_type& m, // vector length
    const real_type_1d_view_type& fn,
    const real_type_1d_view_type& fnr,
    const real_type_1d_view_type& fnp,
    const real_type_1d_view_type& u,
    /* */ real_type& dt)
  {
    const real_type two(2), kr = (-3.0 * _gamma * _gamma + 4.0 * _gamma - 2.0) /
                                 (12.0 * (2.0 - _gamma));
    const real_type one(1), scal1(one / _gamma), scal2(one / (one - _gamma));
    const real_type half(0.5);
    using reducer_value_type = typename Kokkos::Min<real_type>::value_type;
    reducer_value_type alpha;
    Kokkos::Min<real_type> reducer_value(alpha);
    Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(member, m),
      [&](const ordinal_type& i, reducer_value_type& update) {
        /// error estimation
        const real_type abs_est_err = ats<real_type>::abs(
          two * kr * dt *
          (scal1 * fn(i) - scal1 * scal2 * fnr(i) + scal2 * fnp(i)));
        const real_type rel_est_err =
          abs_est_err / (ats<real_type>::abs(u(i)) + ats<real_type>::epsilon());
        const real_type tol_at_i = tol(i,1);

        real_type alpha_at_i(one);
        if (rel_est_err < tol_at_i) {
          if (rel_est_err * two < tol_at_i)
            alpha_at_i = two;
        } else {
          alpha_at_i = half;
        }
        update = update < alpha_at_i ? update : alpha_at_i;
      },
      reducer_value);

    {
      const real_type dtnew = dt * alpha;
      if (alpha < one) {
        dt = dtnew < dtmin ? dtmin : dtnew;
      } else {
        dt = dtnew > dtmax ? dtmax : dtnew;
      }
    }
  }
#endif

  template<typename ProblemType>
  static KOKKOS_INLINE_FUNCTION ordinal_type
  getWorkSpaceSize(const ProblemType& problem)
  {
    const ordinal_type m = problem.getNumberOfEquations();
    const ordinal_type r_val = 1 + m * 8 + m * m;
    return r_val;
  }
};

template<typename ProblemType>
struct TrBDF2_Part1
{
  using problem_type = ProblemType;
  using real_type_1d_view_type = typename problem_type::real_type_1d_view_type;
  using real_type_2d_view_type = typename problem_type::real_type_2d_view_type;

  KOKKOS_DEFAULTED_FUNCTION
  TrBDF2_Part1() = default;

  ProblemType _problem;

  real_type _dt, _gamma;
  real_type_1d_view_type _un, _fn;

  KOKKOS_INLINE_FUNCTION
  ordinal_type getNumberOfTimeODEs() const
  {
    return _problem.getNumberOfTimeODEs();
  }

  KOKKOS_INLINE_FUNCTION
  ordinal_type getNumberOfConstraints() const
  {
    return _problem.getNumberOfConstraints();
  }

  KOKKOS_INLINE_FUNCTION
  ordinal_type getNumberOfEquations() const
  {
    return _problem.getNumberOfEquations();
  }

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION void computeInitValues(
    const MemberType& member,
    const real_type_1d_view_type& u) const
  {
    const ordinal_type m = _problem.getNumberOfEquations();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                         [&](const ordinal_type& i) { u(i) = _un(i); });
    member.team_barrier();
  }

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION void computeJacobian(
    const MemberType& member,
    const real_type_1d_view_type& u,
    const real_type_2d_view_type& J) const
  {
    const real_type one(1), zero(0), half(0.5);
    const ordinal_type m = _problem.getNumberOfTimeODEs(),
                       n = _problem.getNumberOfEquations();

    _problem.computeJacobian(member, u, J);
    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(member, m), [&](const ordinal_type& i) {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, n),
                             [&](const ordinal_type& j) {
                               const auto val = J(i, j);
                               const auto scal = _gamma * _dt * half;
                               J(i, j) = (i == j ? one : zero) - scal * val;
                             });
      });
    member.team_barrier();
  }

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION void computeFunction(
    const MemberType& member,
    const real_type_1d_view_type& u,
    const real_type_1d_view_type& f) const
  {
    const real_type half(0.5);
    const ordinal_type m = _problem.getNumberOfTimeODEs();

    _problem.computeFunction(member, u, f);
    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                         [&](const ordinal_type& i) {
                           const auto val = f(i);
                           const auto scal = _gamma * _dt * half;
                           f(i) = (u(i) - _un(i)) - scal * (val + _fn(i));
                         });

    member.team_barrier();
  }
};

template<typename ProblemType>
struct TrBDF2_Part2
{
  using problem_type = ProblemType;
  using real_type_1d_view_type = typename problem_type::real_type_1d_view_type;
  using real_type_2d_view_type = typename problem_type::real_type_2d_view_type;

  KOKKOS_DEFAULTED_FUNCTION
  TrBDF2_Part2() = default;

  ProblemType _problem;

  real_type _dt, _gamma;
  real_type_1d_view_type _un, _unr;

  KOKKOS_INLINE_FUNCTION
  ordinal_type getNumberOfTimeODEs() const
  {
    return _problem.getNumberOfTimeODEs();
  }

  KOKKOS_INLINE_FUNCTION
  ordinal_type getNumberOfConstraints() const
  {
    return _problem.getNumberOfConstraints();
  }

  KOKKOS_INLINE_FUNCTION
  ordinal_type getNumberOfEquations() const
  {
    return _problem.getNumberOfEquations();
  }

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION void computeInitValues(
    const MemberType& member,
    const real_type_1d_view_type& u) const
  {
    const ordinal_type m = _problem.getNumberOfEquations();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                         [&](const ordinal_type& i) { u(i) = _unr(i); });
  }

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION void computeJacobian(
    const MemberType& member,
    const real_type_1d_view_type& u,
    const real_type_2d_view_type& J) const
  {
    const real_type one(1), two(2), zero(0);
    const ordinal_type m = _problem.getNumberOfTimeODEs(),
                       n = _problem.getNumberOfEquations();

    _problem.computeJacobian(member, u, J);
    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(member, m), [&](const ordinal_type& i) {
        Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(member, n), [&](const ordinal_type& j) {
            const auto val = J(i, j);
            const auto scal = (one - _gamma) / (two - _gamma) * _dt;
            J(i, j) = (i == j ? one : zero) - scal * val;
          });
      });
    member.team_barrier();
  }

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION void computeFunction(
    const MemberType& member,
    const real_type_1d_view_type& u,
    const real_type_1d_view_type& f) const
  {
    const real_type one(1), two(2);
    const ordinal_type m = _problem.getNumberOfTimeODEs();

    _problem.computeFunction(member, u, f);
    Kokkos::parallel_for(
      Kokkos::TeamVectorRange(member, m), [&](const ordinal_type& i) {
        const auto val = f(i);
        const auto scal1 = one / _gamma / (two - _gamma);
        const auto scal2 = scal1 * (one - _gamma) * (one - _gamma);
        const auto scal3 = (one - _gamma) / (two - _gamma) * _dt;
        f(i) = (u(i) - scal1 * _unr(i) + scal2 * _un(i)) - scal3 * val;
      });

    member.team_barrier();
  }
};

} // namespace Impl
} // namespace TChem

#endif
