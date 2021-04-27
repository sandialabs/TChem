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
#ifndef __TCHEM_IMPL_NUM_JACOBIAN2_PFR_HPP__
#define __TCHEM_IMPL_NUM_JACOBIAN2_PFR_HPP__

#include "TChem_Util.hpp"
#include "TChem_Impl_PlugFlowReactor_Problem.hpp"

namespace TChem {
namespace Impl {

struct PlugFlowReactorNumJacobian
{

  template<typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType,
           typename PlugFlowReactorConstDataType>
  KOKKOS_INLINE_FUNCTION static ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf,
    const PlugFlowReactorConstDataType& pfrd)
  {
    using problem_type =
      TChem::Impl::PlugFlowReactor_Problem<KineticModelConstDataType,
                                           KineticSurfModelConstDataType,
                                           PlugFlowReactorConstDataType>;
    problem_type problem;

    return problem_type ::getWorkSpaceSize(kmcd, kmcdSurf);;
  }

  template<typename MemberType,
           typename WorkViewType,
           typename RealType1DViewType,
           typename RealType2DViewType,
           typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType,
           typename PlugFlowReactorConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const RealType1DViewType& x, ///
    /// output
    const RealType2DViewType& Jac,
    const RealType1DViewType& fac, /// numerica jacobian percentage
    // work
    const WorkViewType& work,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf,
    // const input for plug flow reactor
    const PlugFlowReactorConstDataType& pfrd)
  {

    using problem_type =
      TChem::Impl::PlugFlowReactor_Problem<KineticModelConstDataType,
                                           KineticSurfModelConstDataType,
                                           PlugFlowReactorConstDataType>;

    //
    problem_type problem;

    /// problem workspace
    const ordinal_type problem_workspace_size =
      problem_type ::getWorkSpaceSize(kmcd, kmcdSurf);

    auto wptr = work.data();
    auto pw = RealType1DViewType(wptr, problem_workspace_size);
    wptr += problem_workspace_size;

    /// constant values of the problem
    problem._kmcd = kmcd;         // kinetic model
    problem._kmcdSurf = kmcdSurf; // surface kinetic model
    problem._pfrd = pfrd;
    problem._work = pw; // problem workspace array
    problem._fac = fac;    // fac for numerical jacobian

    {
      const ordinal_type m = problem.getNumberOfEquations();
      /// _work is used for evaluating a function
      /// f_0 and f_h should be gained from the tail
      real_type* wptr = problem._work.data() + (problem._work.span() - 2 * m);
      RealType1DViewType f_0(wptr, m);
      wptr += f_0.span();
      RealType1DViewType f_h(wptr, m);
      wptr += f_h.span();

      /// use the default values
      const real_type fac_min(-1), fac_max(-1);
      // NumericalJacobianForwardDifference::team_invoke_detail
      //  (member, problem, fac_min, fac_max, problem._fac, x, f_0, f_h, Jac);
      // NumericalJacobianCentralDifference::team_invoke_detail(
      //   member, problem, fac_min, fac_max, problem._fac, x, f_0, f_h, Jac);
      NumericalJacobianRichardsonExtrapolation::team_invoke_detail
       (member, problem, fac_min, fac_max, problem._fac, x, f_0, f_h, Jac);

    }

    member.team_barrier();
}

};

} // namespace Impl
} // namespace TChem

#endif
