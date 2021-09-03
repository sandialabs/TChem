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
#ifndef __TCHEM_IMPL_SIMPLE_SURFACE_PROBLEM_HPP__
#define __TCHEM_IMPL_SIMPLE_SURFACE_PROBLEM_HPP__

/* This problem advance only surface species.
assumes that gas phase is frozen. In PRF problem, this ODE
is the set to zero (algebraic part of the differential-algebraic
equation (DAE) system).
*/
#include "Tines_Internal.hpp"

#include "TChem_Impl_SurfaceRHS.hpp"
#include "TChem_Util.hpp"


namespace TChem {
namespace Impl {

template<typename ValueType, typename DeviceType>
struct SimpleSurface_Problem
{
  using value_type = ValueType;
  using device_type = DeviceType;
  using scalar_type = typename ats<value_type>::scalar_type;
  using exec_space_type = typename device_type::execution_space;

  using real_type = scalar_type;
  using real_type_0d_view_type = Tines::value_type_0d_view<real_type,device_type>;
  using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;
  using real_type_2d_view_type = Tines::value_type_2d_view<real_type,device_type>;

  /// sacado is value type
  using value_type_0d_view_type = Tines::value_type_0d_view<value_type,device_type>;
  using value_type_1d_view_type = Tines::value_type_1d_view<value_type,device_type>;
  using value_type_2d_view_type = Tines::value_type_2d_view<value_type,device_type>;
  using kinetic_model_type = KineticModelConstData<device_type>;
  using kinetic_surf_model_type = KineticSurfModelConstData<device_type>;

  KOKKOS_DEFAULTED_FUNCTION
  SimpleSurface_Problem() = default;
  /// public access to these member functions
  real_type _t;
  real_type_1d_view_type _Ys;
  real_type _p;
  real_type_1d_view_type _x;
  real_type_1d_view_type _work;
  real_type_1d_view_type _fac; /// numerical jacobian
  kinetic_model_type _kmcd;
  kinetic_surf_model_type _kmcdSurf;
  // ordinal_type _jac_dim;

  KOKKOS_INLINE_FUNCTION
  static ordinal_type getWorkSpaceSize(
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf)
  {
    // const ordinal_type jac_workspace_size =
    //   SurfaceNumJacobian::getWorkSpaceSize(kmcd, kmcdSurf);
    const ordinal_type src_workspace_size =
      SurfaceRHS<real_type,device_type>::getWorkSpaceSize(kmcd, kmcdSurf);

    // const ordinal_type workspace_size = jac_workspace_size > src_workspace_size
    //                                       ? jac_workspace_size
    //                                       : src_workspace_size;

    //
    const ordinal_type jac_workspace_size = 2 * getNumberOfEquations(kmcdSurf);
    const ordinal_type workspace_size = src_workspace_size + jac_workspace_size;

    return workspace_size;
  }

  KOKKOS_INLINE_FUNCTION
  static ordinal_type getNumberOfTimeODEs(
    const kinetic_surf_model_type& kmcdSurf)
  {
    return kmcdSurf.nSpec;
  }

  KOKKOS_INLINE_FUNCTION
  static ordinal_type getNumberOfConstraints(
    const kinetic_surf_model_type& kmcdSurf)
  {
    return 0;
  }

  KOKKOS_INLINE_FUNCTION
  static ordinal_type getNumberOfEquations(
    const kinetic_surf_model_type& kmcdSurf)
  {
    return getNumberOfTimeODEs(kmcdSurf) + getNumberOfConstraints(kmcdSurf);
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
    return getNumberOfTimeODEs(_kmcdSurf);
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

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION void computeInitValues(
    const MemberType& member,
    const real_type_1d_view_type& x) const
  {
    /// this is probably not used
    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, x.extent(0)),
                         [&](const ordinal_type& i) { x(i) = _x(i); });
    member.team_barrier();
  }

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION void computeJacobian(const MemberType& member,
                                              const real_type_1d_view_type& x,
                                              const real_type_2d_view_type& J) const
  {
    // Jac only for surface phase
    // Impl::SurfaceNumJacobian::team_invoke(
    //   member, _t, _Ys, x, _p, J, _work, _kmcd, _kmcdSurf);
    // member.team_barrier();

    const ordinal_type m = getNumberOfEquations();
    /// _work is used for evaluating a function
    /// f_0 and f_h should be gained from the tail
    real_type* wptr = _work.data() + (_work.span() - 2 * m);
    real_type_1d_view_type f_0(wptr, m);
    wptr += f_0.span();
    real_type_1d_view_type f_h(wptr, m);
    wptr += f_h.span();

    /// use the default values
    const real_type fac_min(-1), fac_max(-1);
    Tines::NumericalJacobianForwardDifference<value_type, device_type>::invoke(
          member, *this, fac_min, fac_max, _fac, x, f_0, f_h, J);
    // NumericalJacobianCentralDifference::team_invoke_detail(
    //   member, *this, fac_min, fac_max, _fac, x, f_0, f_h, J);
    // NumericalJacobianRichardsonExtrapolation::team_invoke_detail
    //  (member, *this, fac_min, fac_max, _fac, x, f_0, f_h, J);

  }

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION void computeFunction(const MemberType& member,
                                              const real_type_1d_view_type& x,
                                              const real_type_1d_view_type& f) const
  {

    Impl::SurfaceRHS<real_type, device_type>::team_invoke(
      member, _t, _Ys, x, _p, f, _work, _kmcd, _kmcdSurf);
    member.team_barrier();
  }
};

} // namespace Impl
} // namespace TChem

#endif
