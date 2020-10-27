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
#ifndef __TCHEM_IMPL_IGNITION_ZEROD_PROBLEM_HPP__
#define __TCHEM_IMPL_IGNITION_ZEROD_PROBLEM_HPP__

#include "TChem_Util.hpp"

#include "TChem_Impl_JacobianReduced.hpp"
#include "TChem_Impl_SourceTerm.hpp"

#include "TChem_Impl_NumericalJacobianCentralDifference.hpp"
#include "TChem_Impl_NumericalJacobianForwardDifference.hpp"
#include "TChem_Impl_NumericalJacobianRichardsonExtrapolation.hpp"

namespace TChem {
namespace Impl {

template<typename KineticModelConstDataType>
struct IgnitionZeroD_Problem
{
  using kmcd_type = KineticModelConstDataType;
  using exec_space_type = typename kmcd_type::exec_space_type;
  using real_type_1d_view_type = typename kmcd_type::real_type_1d_view_type;
  using real_type_2d_view_type = typename kmcd_type::real_type_2d_view_type;

  KOKKOS_DEFAULTED_FUNCTION
  IgnitionZeroD_Problem() = default;

  /// public access to these member functions
  real_type _p;
  real_type_1d_view_type _x;
  real_type_1d_view_type _fac; /// numerical jacobian
  real_type_1d_view_type _work;
  kmcd_type _kmcd;

  KOKKOS_INLINE_FUNCTION
  static ordinal_type getNumberOfTimeODEs(const KineticModelConstDataType& kmcd)
  {
    return kmcd.nSpec + 1;
  }

  KOKKOS_INLINE_FUNCTION
  static ordinal_type getNumberOfConstraints(
    const KineticModelConstDataType& kmcd)
  {
    return 0;
  }

  KOKKOS_INLINE_FUNCTION
  static ordinal_type getNumberOfEquations(
    const KineticModelConstDataType& kmcd)
  {
    return getNumberOfTimeODEs(kmcd) + getNumberOfConstraints(kmcd);
  }

  KOKKOS_INLINE_FUNCTION
  static ordinal_type getWorkSpaceSize(const KineticModelConstDataType& kmcd)
  {
    const ordinal_type src_workspace_size = SourceTerm::getWorkSpaceSize(kmcd);
#if defined(TCHEM_ENABLE_PROBLEMS_NUMERICAL_JACOBIAN)
    const ordinal_type jac_workspace_size = 2 * getNumberOfEquations(kmcd);
    const ordinal_type workspace_size = src_workspace_size + jac_workspace_size;
#else
    const ordinal_type jac_workspace_size =
      JacobianReduced::getWorkSpaceSize(kmcd);
    const ordinal_type workspace_size =
      (jac_workspace_size > src_workspace_size ? jac_workspace_size
                                               : src_workspace_size);
#endif
    return workspace_size;
  }

  ///
  /// non static functions that require the object
  /// workspace size is required without kmcd
  ///
  KOKKOS_INLINE_FUNCTION
  ordinal_type getNumberOfTimeODEs() const
  {
    return getNumberOfTimeODEs(_kmcd);
  }

  KOKKOS_INLINE_FUNCTION
  ordinal_type getNumberOfConstraints() const
  {
    return getNumberOfConstraints(_kmcd);
  }

  KOKKOS_INLINE_FUNCTION
  ordinal_type getNumberOfEquations() const
  {
    return getNumberOfTimeODEs() + getNumberOfConstraints();
  }

  KOKKOS_INLINE_FUNCTION
  ordinal_type getWorkSpaceSize() const { return getWorkSpaceSize(_kmcd); }

  template<typename MemberType, typename RealType1DViewType>
  KOKKOS_INLINE_FUNCTION void computeInitValues(
    const MemberType& member,
    const RealType1DViewType& x) const
  {
    /// this is probably not used
    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, x.extent(0)),
                         [&](const ordinal_type& i) { x(i) = _x(i); });
    member.team_barrier();
  }

  template<typename MemberType, typename RealType1DViewType>
  KOKKOS_INLINE_FUNCTION void computeFunction(const MemberType& member,
                                              const RealType1DViewType& x,
                                              const RealType1DViewType& f) const
  {
    const real_type t = x(0);
    const real_type_1d_view_type Ys(&x(1), _kmcd.nSpec);
    Impl::SourceTerm::team_invoke(member, t, _p, Ys, f, _work, _kmcd);
    member.team_barrier();
  }

  template<typename MemberType,
           typename RealType1DViewType,
           typename RealType2DViewType>
  KOKKOS_INLINE_FUNCTION void computeJacobian(const MemberType& member,
                                              const RealType1DViewType& x,
                                              const RealType2DViewType& J) const
  {
#if defined(TCHEM_ENABLE_PROBLEMS_NUMERICAL_JACOBIAN)
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
#else
    const real_type t = x(0);
    const real_type_1d_view_type Ys(&x(1), _kmcd.nSpec);
    Impl::JacobianReduced::team_invoke(member, t, _p, Ys, J, _work, _kmcd);
    member.team_barrier();
#endif
  }
};

} // namespace Impl
} // namespace TChem

#endif
