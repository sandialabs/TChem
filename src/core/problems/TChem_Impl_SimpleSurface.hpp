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
#ifndef __TCHEM_IMPL_SIMPLESURFACE_HPP__
#define __TCHEM_IMPL_SIMPLESURFACE_HPP__

#include "TChem_Util.hpp"

#include "TChem_Impl_SimpleSurface_Problem.hpp"
#include "TChem_Impl_TimeIntegrator.hpp"

namespace TChem {
namespace Impl {

struct SimpleSurface
{

  template<typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf)
  {

    //
    using problem_type =
      TChem::Impl::SimpleSurface_Problem<KineticModelConstDataType,
                                         KineticSurfModelConstDataType>;
    problem_type problem;
    problem._kmcd = kmcd;
    problem._kmcdSurf = kmcdSurf;
    return TimeIntegrator::getWorkSpaceSize(problem) +
           problem.getNumberOfEquations(kmcdSurf); /// temporal vector tolerence
  }

  template<typename MemberType,
           typename WorkViewType,
           typename RealType0DViewType,
           typename RealType1DViewType,
           typename RealType2DViewType,
           typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
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
    const RealType1DViewType& vals,
    /// output (final output conditions)
    const RealType0DViewType& t_out,
    const RealType0DViewType& dt_out,
    const RealType1DViewType& vals_out,
    // const values
    const real_type& temperature, /// temperature
    const real_type& pressure,    /// pressure
    const RealType1DViewType& Ys, /// mass fraction (kmcd.nSpec)
    /// workspace
    const WorkViewType& work,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf)
  {
    /// all tolerence are from users; for this, it should be set in the front
    /// interface
    /// const real_type atol_newton = 1e-6, rtol_newton = 1e-5, tol_time = 1e-4;

    using problem_type =
      TChem::Impl::SimpleSurface_Problem<KineticModelConstDataType,
                                         KineticSurfModelConstDataType>;
    using real_type_1d_view_type =
      typename problem_type::real_type_1d_view_type;

    problem_type problem;

    /// problem workspace
    const ordinal_type problem_workspace_size =
      problem_type::getWorkSpaceSize(kmcd, kmcdSurf);

    auto wptr = work.data();
    auto pw = real_type_1d_view_type(wptr, problem_workspace_size);
    wptr += pw.span();

    /// error check
    const ordinal_type workspace_used(wptr - work.data()),
      workspace_extent(work.extent(0));
    if (workspace_used > workspace_extent) {
      Kokkos::abort("Error: workspace used is larger than it is provided\n");
    }
    /// time integrator workspace
    auto tw = WorkViewType(wptr, workspace_extent - workspace_used);

    /// constant values of the problem
    problem._p = pressure;        // pressure
    problem._kmcd = kmcd;         // kinetic model
    problem._kmcdSurf = kmcdSurf; // surface kinetic model
    problem._Ys = Ys;             // mass fraction is constant
    problem._t = temperature;     // temperature
    problem._work = pw;           // problem workspace array
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
