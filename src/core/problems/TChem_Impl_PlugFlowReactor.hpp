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
#ifndef __TCHEM_IMPL_PLUGFLOWREACTOR_HPP__
#define __TCHEM_IMPL_PLUGFLOWREACTOR_HPP__

#include "TChem_Util.hpp"

#include "TChem_Impl_PlugFlowReactor_Problem.hpp"
#include "TChem_Impl_TimeIntegrator.hpp"

namespace TChem {
namespace Impl {

struct PlugFlowReactor
{

  template<typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType,
           typename PlugFlowReactorConstDataType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf,
    const PlugFlowReactorConstDataType& pfrd)
  {
    //
    using problem_type =
      TChem::Impl::PlugFlowReactor_Problem<KineticModelConstDataType,
                                           KineticSurfModelConstDataType,
                                           PlugFlowReactorConstDataType>;

    problem_type problem;
    problem._kmcd = kmcd;
    problem._kmcdSurf = kmcdSurf;
    return TimeIntegrator::getWorkSpaceSize(problem);
  }

  template<typename MemberType,
           typename WorkViewType,
           typename RealType0DViewType,
           typename RealType1DViewType,
           typename RealType2DViewType,
           typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType,
           typename PlugFlowReactorConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input iteration and qoi index to store
    const ordinal_type& max_num_newton_iterations,
    const ordinal_type& max_num_time_iterations,
    /// input time step and time range
    const RealType1DViewType& tol_newton,
    const RealType2DViewType& tol_time,
    const RealType1DViewType& fac, /// numerica jacobian percentage
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
    const WorkViewType& work,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf,
    const PlugFlowReactorConstDataType& pfrd)
  {
    // printf("at Impl::PlugFlowReactor\n");
    // for (ordinal_type k = 0; k < vals.extent(0); k++) {
    //     printf(" k %d, vals %e  \n",k, vals(k) );
    // }

    // const real_type atol_newton = 1e-6, rtol_newton = 1e-5, tol_time = 1e-4;

    using problem_type =
      TChem::Impl::PlugFlowReactor_Problem<KineticModelConstDataType,
                                           KineticSurfModelConstDataType,
                                           PlugFlowReactorConstDataType>;
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
    auto tw = WorkViewType(wptr, workspace_extent - workspace_used);

    /// constant values of the problem
    problem._kmcd = kmcd;         // kinetic model
    problem._kmcdSurf = kmcdSurf; // surface kinetic model
    problem._pfrd = pfrd;
    problem._work = pw; // problem workspace array
    problem._fac = fac;    // fac for numerical jacobian

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
  }
};

} // namespace Impl
} // namespace TChem

#endif
