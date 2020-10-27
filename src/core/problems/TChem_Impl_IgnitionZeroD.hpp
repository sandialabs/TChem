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
#ifndef __TCHEM_IMPL_IGNITION_ZEROD_HPP__
#define __TCHEM_IMPL_IGNITION_ZEROD_HPP__

#include "TChem_Util.hpp"

#include "TChem_Impl_IgnitionZeroD_Problem.hpp"
#include "TChem_Impl_TimeIntegrator.hpp"

namespace TChem {
namespace Impl {

struct IgnitionZeroD
{
  template<typename KineticModelConstDataType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd)
  {
    using problem_type =
      TChem::Impl::IgnitionZeroD_Problem<KineticModelConstDataType>;
    problem_type problem;
    problem._kmcd = kmcd;

    return TimeIntegrator::getWorkSpaceSize(problem);
  }

  template<typename MemberType,
           typename WorkViewType,
           typename RealType0DViewType,
           typename RealType1DViewType,
           typename RealType2DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    /// input iteration and qoi index to store
    const ordinal_type& max_num_newton_iterations,
    const ordinal_type& max_num_time_iterations,
    const RealType1DViewType& tol_newton,
    const RealType2DViewType& tol_time,
    const RealType1DViewType& fac, /// numerica jacobian percentage
    /// input time step and time range
    const real_type& dt_in,
    const real_type& dt_min,
    const real_type& dt_max,
    const real_type& t_beg,
    const real_type& t_end,
    /// input (initial condition)
    const real_type& pressure,      /// pressure
    const RealType1DViewType& vals, /// mass fraction (kmcd.nSpec)
    /// output (final output conditions)
    const RealType0DViewType& t_out,
    const RealType0DViewType& dt_out,
    const RealType0DViewType& pressure_out,
    const RealType1DViewType& vals_out,
    /// workspace
    const WorkViewType& work,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    using problem_type =
      TChem::Impl::IgnitionZeroD_Problem<KineticModelConstDataType>;
    problem_type problem;

    /// problem workspace
    const ordinal_type problem_workspace_size =
      problem_type::getWorkSpaceSize(kmcd);
    auto wptr = work.data();
    auto pw = typename problem_type::real_type_1d_view_type(
      wptr, problem_workspace_size);
    wptr += problem_workspace_size;

    /// error check
    const ordinal_type workspace_used(wptr - work.data()),
      workspace_extent(work.extent(0));
    if (workspace_used > workspace_extent) {
      Kokkos::abort("Error: workspace used is larger than it is provided\n");
    }

    /// time integrator workspace
    auto tw = WorkViewType(wptr, workspace_extent - workspace_used);

    /// initialize problem
    problem._p = pressure; // pressure
    problem._work = pw;    // problem workspace array
    problem._kmcd = kmcd;  // kinetic model
    problem._fac = fac;    // fac for numerical jacobian

    const ordinal_type r_val =
      TimeIntegrator::team_invoke_detail(member,
                                         problem,
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

    /// pressure is constant, make sure it in the next restarting iteration
    Kokkos::single(Kokkos::PerTeam(member), [=]() {
      const real_type zero(0);
      if (r_val == 0) {
        pressure_out() = pressure;
      } else {
        pressure_out() = zero;
      }
    });
  }

  template<typename MemberType,
           typename WorkViewType,
           typename RealType0DViewType,
           typename RealType1DViewType,
           typename RealType2DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input iteration and qoi index to store
    const ordinal_type& max_num_newton_iterations,
    const ordinal_type& max_num_time_iterations,
    const RealType1DViewType& tol_newton,
    const RealType2DViewType& tol_time,
    const RealType1DViewType& fac,
    /// input time step and time range
    const real_type& dt_in,
    const real_type& dt_min,
    const real_type& dt_max,
    const real_type& t_beg,
    const real_type& t_end,
    /// input (initial condition)
    const real_type& pressure,      /// pressure
    const RealType1DViewType& vals, /// temperature, mass fractions
    /// output (final output conditions)
    const RealType0DViewType& t_out,
    const RealType0DViewType& dt_out,
    const RealType0DViewType& pressure_out,
    const RealType1DViewType& vals_out,
    /// workspace
    const WorkViewType& work,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    // const real_type atol_newton = 1e-10, rtol_newton = 1e-6, tol_time = 1e-4;
    team_invoke_detail(member,
                       max_num_newton_iterations,
                       max_num_time_iterations,
                       tol_newton,
                       tol_time,
                       fac,
                       dt_in,
                       dt_min,
                       dt_max,
                       t_beg,
                       t_end,
                       pressure,
                       vals,
                       t_out,
                       dt_out,
                       pressure_out,
                       vals_out,
                       work,
                       kmcd);
    member.team_barrier();
    /// input is valid and output is not valid then, send warning message
    const real_type zero(0);
    if (dt_in > zero && dt_out() < zero) {
      Kokkos::single(Kokkos::PerTeam(member), [&]() {
        printf("Warning: IgnitionZeroD sample id(%d) failed\n",
               int(member.league_rank()));
      });
    }
  }
};

} // namespace Impl
} // namespace TChem

#endif
