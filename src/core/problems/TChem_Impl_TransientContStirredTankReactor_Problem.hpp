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
#ifndef __TCHEM_IMPL_TRANSIENT_CONT_STIRRED_TANK_REACTOR_PROBLEM_HPP__
#define __TCHEM_IMPL_TRANSIENT_CONT_STIRRED_TANK_REACTOR_PROBLEM_HPP__

/* This problem advance a cstr with surface reactions.
 */
#include "TChem_Impl_NumericalJacobianCentralDifference.hpp"
#include "TChem_Impl_NumericalJacobianForwardDifference.hpp"
#include "TChem_Impl_NumericalJacobianRichardsonExtrapolation.hpp"
#include "TChem_Impl_MolarWeights.hpp"
#include "TChem_Impl_TransientContStirredTankReactorRHS.hpp"
#include "TChem_Util.hpp"

namespace TChem {
namespace Impl {

template<typename KineticModelConstDataType,
         typename KineticSurfModelConstDataType,
         typename ContStirredTankReactorConstDataType>
struct TransientContStirredTankReactor_Problem
{
  using kmcd_type = KineticModelConstDataType;
  using exec_space_type = typename kmcd_type::exec_space_type;
  using real_type_1d_view_type = typename kmcd_type::real_type_1d_view_type;
  using real_type_2d_view_type = typename kmcd_type::real_type_2d_view_type;

  KOKKOS_DEFAULTED_FUNCTION
  TransientContStirredTankReactor_Problem() = default;
  /// public access to these member functions
  real_type_1d_view _x;
  real_type_1d_view _work;
  real_type_1d_view_type _fac; /// numerical jacobian
  KineticModelConstDataType _kmcd;
  KineticSurfModelConstDataType _kmcdSurf;
  ContStirredTankReactorConstDataType _cstr;

  KOKKOS_INLINE_FUNCTION
  static ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf)
  {
    const ordinal_type jac_workspace_size = 2 * getNumberOfEquations(kmcd, kmcdSurf);
    const ordinal_type src_workspace_size = TransientContStirredTankReactorRHS
    ::getWorkSpaceSize(kmcd, kmcdSurf);//
    const ordinal_type workspace_size = src_workspace_size + jac_workspace_size;

    return workspace_size;
  }

  KOKKOS_INLINE_FUNCTION
  static ordinal_type getNumberOfTimeODEs(const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf)
  {
#if defined(TCHEM_ENABLE_PROBLEM_DAE_CSTR)
    return kmcd.nSpec +  1;
#else
    return kmcd.nSpec +  kmcdSurf.nSpec + 1;
#endif
  }

  KOKKOS_INLINE_FUNCTION
  static ordinal_type getNumberOfConstraints(
    const KineticSurfModelConstDataType& kmcdSurf)
  {
#if defined(TCHEM_ENABLE_PROBLEM_DAE_CSTR)
    return kmcdSurf.nSpec ; //
#else
    return 0;
#endif
  }

  KOKKOS_INLINE_FUNCTION
  static ordinal_type getNumberOfEquations(
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf)
  {
    return getNumberOfTimeODEs(kmcd, kmcdSurf) + getNumberOfConstraints(kmcdSurf);
  }

  ///
  /// non static functions that require the object
  /// workspace size is required without kmcd
  ///
  KOKKOS_INLINE_FUNCTION
  ordinal_type getWorkSpaceSize() const
  {
    return getWorkSpaceSize(_kmcd, _kmcdSurf);
  }

  KOKKOS_INLINE_FUNCTION
  ordinal_type getNumberOfTimeODEs() const
  {
    return getNumberOfTimeODEs(_kmcd, _kmcdSurf);
  }

  KOKKOS_INLINE_FUNCTION
  ordinal_type getNumberOfConstraints() const
  {
    return getNumberOfConstraints(_kmcdSurf);
  }

  KOKKOS_INLINE_FUNCTION
  ordinal_type getNumberOfEquations() const
  {
    return getNumberOfTimeODEs() + getNumberOfConstraints();
  }

  template<typename MemberType, typename RealType1DViewType>
  KOKKOS_INLINE_FUNCTION void computeInitValues(
    const MemberType& member,
    const RealType1DViewType& x) const
  {
    /// this is probably not used
    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, x.extent(0)),
                         [&](const ordinal_type& i) { x(i) = _x(i); });

    // compute constraint and store value to be use in function and jacobian
    // resolve a non linear system

    member.team_barrier();
  }

  template<typename MemberType,
           typename RealType1DViewType,
           typename RealType2DViewType>
  KOKKOS_INLINE_FUNCTION void computeJacobian(const MemberType& member,
                                              const RealType1DViewType& x,
                                              const RealType2DViewType& J) const
  {

    const ordinal_type m = getNumberOfEquations();
    /// _work is used for evaluating a function
    /// f_0 and f_h should be gained from the tail
    real_type* wptr = _work.data() + (_work.span() - 2 * m);
    RealType1DViewType f_0(wptr, m);
    wptr += f_0.span();
    RealType1DViewType f_h(wptr, m);
    wptr += f_h.span();

    /// use the default values
    const real_type fac_min(-1), fac_max(-1);
    // NumericalJacobianForwardDifference::team_invoke_detail
    //  (member, *this, fac_min, fac_max, _fac, x, f_0, f_h, J);
    NumericalJacobianCentralDifference::team_invoke_detail(
      member, *this, fac_min, fac_max, _fac, x, f_0, f_h, J);
    // NumericalJacobianRichardsonExtrapolation::team_invoke_detail
    //  (member, *this, fac_min, fac_max, _fac, x, f_0, f_h, J);

//
#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)

#endif
  }

  template<typename MemberType, typename RealType1DViewType>
  KOKKOS_INLINE_FUNCTION void computeFunction(const MemberType& member,
                                              const RealType1DViewType& x,
                                              const RealType1DViewType& f) const
  {
    const real_type t = x(0);
    const real_type_1d_view Ys(&x(1), _kmcd.nSpec);
    const real_type_1d_view siteFraction(&x(_kmcd.nSpec+1), _kmcdSurf.nSpec);
    //compute density
    const real_type Wmix = MolarWeights::team_invoke(member, Ys, _kmcd);
    // compute pressure
    const real_type density = _cstr.pressure * Wmix/ _kmcd.Runiv / t ;

    Impl::TransientContStirredTankReactorRHS ::team_invoke(member,
                                           t, // constant temperature
                                           Ys,
                                           siteFraction,
                                           density,
                                           _cstr.pressure, // constant pressure
                                           f,
                                           _work,
                                           _kmcd,
                                           _kmcdSurf,
                                           _cstr);
//
#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen(
        "TransientContStirredTankReactor_Problem_computeFunction.team_invoke.test.out", "a+");
      fprintf(fs, ":::: input\n");
      fprintf(fs,
              "     nSpec %3d, nReac %3d, site density %e\n",
              _kmcdSurf.nSpec,
              _kmcdSurf.nReac,
              _kmcdSurf.sitedensity);

      for (int i = 0; i < Ys.extent(0); ++i)
        fprintf(fs, "   i %3d,  Ys %e, \n", i, Ys(i));

      for (int i = 0; i < siteFraction.extent(0); ++i)
        fprintf(fs, "   i %3d,  siteFraction %e, \n", i, siteFraction(i));

      for (int i = 0; i < x.extent(0); ++i)
        fprintf(fs, "   i %3d,  x %e, \n", i, x(i));

      fprintf(fs, ":::: output\n");
      for (int i = 0; i < f.extent(0); ++i)
        fprintf(fs, "   i %3d,  f %e, \n", i, f(i));
    }
#endif
  }
};

} // namespace Impl
} // namespace TChem

#endif
