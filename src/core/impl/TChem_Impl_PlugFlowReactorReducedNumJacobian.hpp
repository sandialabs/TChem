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
#ifndef __TCHEM_IMPL_RED_JACOBIAN_PFR_TERM_HPP__
#define __TCHEM_IMPL_RED_JACOBIAN_PFR_TERM_HPP__

#include "TChem_Util.hpp"
#include "TChem_Impl_PlugFlowReactor_Problem.hpp"
#include "TChem_Impl_NewtonSolver.hpp"

namespace TChem {
namespace Impl {

struct PlugFlowReactorReducedNumJacobian
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
    /// problem workspace
    const ordinal_type problem_workspace_size =
      problem_type ::getWorkSpaceSize(kmcd, kmcdSurf);

    const auto nTotalEqns = kmcd.nSpec + kmcdSurf.nSpec + 3;
    const auto nEqns = kmcd.nSpec + 3;

    const auto jacComp_workspace_size =  nTotalEqns*nTotalEqns + nEqns*nEqns +
     3*nEqns*kmcdSurf.nSpec + kmcdSurf.nSpec*kmcdSurf.nSpec;

     const ordinal_type m = kmcdSurf.nSpec, n = m;
     const ordinal_type WSNewton =m * n + n * n + 2*(m < n ? m : n) + n * nEqns;

    return problem_workspace_size + jacComp_workspace_size + WSNewton;
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
    const RealType2DViewType& redJac,
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

    auto nTotalEqns = kmcd.nSpec + kmcdSurf.nSpec + 3;

    auto Jac  = RealType2DViewType(wptr, nTotalEqns, nTotalEqns);
    wptr+=nTotalEqns*nTotalEqns;


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

    auto nEqns = kmcd.nSpec + 3;
    auto gu   = RealType2DViewType(wptr, nEqns, nEqns);
    wptr+=nEqns*nEqns;
    auto gv   = RealType2DViewType(wptr, nEqns, kmcdSurf.nSpec);
    wptr+=nEqns*kmcdSurf.nSpec;
    auto fu   = RealType2DViewType(wptr, kmcdSurf.nSpec, nEqns);
    wptr+=nEqns*kmcdSurf.nSpec;
    auto fv   = RealType2DViewType(wptr, kmcdSurf.nSpec, kmcdSurf.nSpec);
    wptr+=kmcdSurf.nSpec*kmcdSurf.nSpec;
    auto vu   = RealType2DViewType(wptr, kmcdSurf.nSpec, nEqns);
    wptr+=nEqns*kmcdSurf.nSpec;

    //check this matrix A size  kmcdSurf.nSpecXkmcdSurf.nSpec, but dx is kmcdSurf.nSpec X nEq
    const ordinal_type m = nEqns, n = m;
    const ordinal_type WSNewton = m * m + n * n + n + (m < n ? m : n) + n + n;
    wptr+=WSNewton;

    auto workNewtonSolver = RealType1DViewType(wptr, WSNewton);

    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(member, kmcd.nSpec + 3), [&](const ordinal_type& k) {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, kmcd.nSpec + 3),
          [&](const ordinal_type& i) {
            gu(k , i) = Jac(k,i);
        });
    });

    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(member, kmcd.nSpec + 3), [&](const ordinal_type& k) {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, kmcdSurf.nSpec),
          [&](const ordinal_type& i) {
            gv(k , i) = Jac(k,i + kmcd.nSpec + 3);
        });
    });

    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(member, kmcdSurf.nSpec), [&](const ordinal_type& k) {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, kmcd.nSpec + 3),
          [&](const ordinal_type& i) {
            //negative sign is needed, check notes on DAE
            fu(k , i) = -Jac(k + kmcd.nSpec + 3, i );
        });
    });

    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(member, kmcdSurf.nSpec), [&](const ordinal_type& k) {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, kmcdSurf.nSpec),
          [&](const ordinal_type& i) {
            fv(k , i) = Jac(k + kmcd.nSpec + 3, i+ kmcd.nSpec + 3 );
        });
    });

    member.team_barrier();

    bool is_valid(true);
    /// sanity check
    Tines::CheckNanInf::invoke(member, fv, is_valid);

    if (is_valid) {
      // solve linear system Ax= B A(Fv) B(Fu) X(dvdu)
        /// solve the equation: dx = -J^{-1} f(x);
        // dx = vu J = fv f=fu
      ordinal_type matrix_rank(0);
      Tines::SolveLinearSystem
	::invoke(member, fv, vu, fu, workNewtonSolver, matrix_rank);
    } else{
      printf("Jacabian has Nan or Inf \n");
    }

    member.team_barrier();

    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(member, kmcd.nSpec + 3),
      [&](const ordinal_type &i) {
      Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(member, kmcd.nSpec + 3),
        [&](const ordinal_type &j) {
          real_type val(0);
          for (ordinal_type k = 0; k < kmcdSurf.nSpec; k++) {
            val +=  gv(i,k)*vu(k,j);
          }
          redJac(i,j) = gu(i,j) + val   ;//
      });
    });

    member.team_barrier();


#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
    }
#endif

}

};

} // namespace Impl
} // namespace TChem

#endif
