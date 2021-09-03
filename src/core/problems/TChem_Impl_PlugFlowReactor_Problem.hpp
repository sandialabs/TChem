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
#ifndef __TCHEM_IMPL_PLUG_FLOW_REACTOR_PROBLEM_HPP__
#define __TCHEM_IMPL_PLUG_FLOW_REACTOR_PROBLEM_HPP__

#include "Tines_Internal.hpp"

#include "TChem_KineticModelData.hpp"
#include "TChem_Impl_PlugFlowReactorRHS.hpp"

namespace TChem {
namespace Impl {

  template<typename ValueType, typename DeviceType>
  struct PlugFlowReactor_Problem
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
    using pdf_model_type = PlugFlowReactorData;
    // static_assert(ats<value_type>::is_sacado, "This problem must be templated with Sacado SLFad");

    real_type_1d_view_type _work;
    real_type_1d_view_type _fac;
    kinetic_model_type _kmcd;
    kinetic_surf_model_type _kmcdSurf;
    pdf_model_type _pfrd;

    KOKKOS_DEFAULTED_FUNCTION
    PlugFlowReactor_Problem() = default;

    KOKKOS_INLINE_FUNCTION
    static ordinal_type getNumberOfTimeODEs(const kinetic_model_type& kmcd)
    {
      return kmcd.nSpec + 3; // Temp, mass fraction, density, velocity
    }

    KOKKOS_INLINE_FUNCTION
    static ordinal_type getNumberOfConstraints(
      const kinetic_surf_model_type& kmcdSurf)
    {
      return kmcdSurf.nSpec; // site fraction
    }

    KOKKOS_INLINE_FUNCTION
    static ordinal_type getNumberOfEquations(
      const kinetic_model_type& kmcd,
      const kinetic_surf_model_type& kmcdSurf)
    {
      return getNumberOfTimeODEs(kmcd) + getNumberOfConstraints(kmcdSurf);
    }


    KOKKOS_INLINE_FUNCTION
    ordinal_type getNumberOfTimeODEs() const
    {
      return getNumberOfTimeODEs(_kmcd);
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

    KOKKOS_INLINE_FUNCTION
    ordinal_type getWorkSpaceSize() const
    {
      return getWorkSpaceSize(_kmcd, _kmcdSurf);
    }

    KOKKOS_INLINE_FUNCTION
    static ordinal_type getWorkSpaceSize(const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf)
    {

      const ordinal_type m = getNumberOfEquations(kmcd, kmcdSurf);
      const ordinal_type wlen = PlugFlowReactorRHS<value_type, device_type>
      ::getWorkSpaceSize(kmcd, kmcdSurf ) + 2*m;

      return wlen*ats<value_type>::sacadoStorageCapacity();
    }

    KOKKOS_INLINE_FUNCTION
    void setWorkspace(const real_type_1d_view_type& work)
    {
      _work = work;
    }

    template<typename MemberType>
    KOKKOS_INLINE_FUNCTION void computeFunction(const MemberType& member,
                                                const real_type_1d_view_type& x,
                                                const real_type_1d_view_type& f) const
    {
      const real_type t = x(0);
      const real_type_1d_view_type Ys(&x(1), _kmcd.nSpec);
      const real_type density = x(_kmcd.nSpec + 1);
      const real_type vel = x(_kmcd.nSpec + 2);
      const real_type_1d_view_type siteFraction(&x(_kmcd.nSpec + 3), _kmcdSurf.nSpec);
      const real_type Wmix = MolarWeights<real_type, device_type>::team_invoke(member, Ys, _kmcd);
      const real_type p = _kmcd.Runiv * t * density / Wmix; // compute pressure

      PlugFlowReactorRHS<real_type, device_type> ::team_invoke(member,
                                             t,
                                             Ys,
                                             siteFraction,
                                             density,
                                             p,
                                             vel,
                                             f,
                                             _work,
                                             _kmcd,
                                             _kmcdSurf,
                                             _pfrd);
      member.team_barrier();
    }

    template<typename MemberType>
    KOKKOS_INLINE_FUNCTION
    void computeFunctionSacado(const MemberType& member,
			       const value_type_1d_view_type& x,
			       const value_type_1d_view_type& f) const
    {
      const value_type t = x(0);
      using range_type = Kokkos::pair<ordinal_type, ordinal_type>;
      const value_type_1d_view_type Ys = Kokkos::subview(x,range_type(1, _kmcd.nSpec + 1  ));
      const value_type density = x(_kmcd.nSpec + 1);
      const value_type vel = x(_kmcd.nSpec + 2);

      const value_type Wmix = MolarWeights<value_type, device_type>
      ::team_invoke(member, Ys, _kmcd);
      const value_type p = _kmcd.Runiv * t * density / Wmix; // compute pressure

      const value_type_1d_view_type siteFraction =
      Kokkos::subview(x,range_type(_kmcd.nSpec + 3, _kmcd.nSpec + 3 +_kmcdSurf.nSpec  ));

      PlugFlowReactorRHS<value_type, device_type>::team_invoke_sacado(member,
                                             t,
                                             Ys,
                                             siteFraction,
                                             density,
                                             p,
                                             vel,
                                             f,
                                             _work,
                                             _kmcd,
                                             _kmcdSurf,
                                             _pfrd);

      member.team_barrier();
    }

    /// this one is used in time integration nonlinear solve
    template<typename MemberType>
    KOKKOS_INLINE_FUNCTION
    void computeSacadoJacobian(const MemberType& member,
				       const real_type_1d_view_type& s,
				       const real_type_2d_view_type& J) const
    {
      const ordinal_type len = ats<value_type>::sacadoStorageCapacity();
      const ordinal_type m = getNumberOfEquations();

      real_type* wptr = _work.data() + (_work.span() - 2*m*len );
      value_type_1d_view_type x(wptr, m, m+1); wptr += m*len;
      value_type_1d_view_type f(wptr, m, m+1); wptr += m*len;

      Kokkos::parallel_for
	(Kokkos::TeamVectorRange(member, m),
	 [=](const int &i) {
	   x(i) = value_type(m, i, s(i));
	 });
      member.team_barrier();
      computeFunctionSacado(member, x, f);
      member.team_barrier();
      Kokkos::parallel_for
	(Kokkos::TeamThreadRange(member, m),
	 [=](const int &i) {
	   Kokkos::parallel_for
	     (Kokkos::ThreadVectorRange(member, m),
	      [=](const int &j) {
	       J(i,j) = f(i).fastAccessDx(j);
	     });
	 });
      member.team_barrier();
    }

  //
  /// this one is used in time integration nonlinear solve
  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION
  void computeJacobian(const MemberType& member,
             const real_type_1d_view_type& s,
             const real_type_2d_view_type& J) const
  {

#if defined(TCHEM_ENABLE_SACADO_JACOBIAN_PLUG_FLOW_REACTOR)
     computeSacadoJacobian(member, s, J);
#else
     computeNumericalJacobian(member, s, J);
#endif

  }

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION void computeNumericalJacobian(const MemberType& member,
                                              const real_type_1d_view_type& x,
                                              const real_type_2d_view_type& J) const
{
  const ordinal_type m = getNumberOfEquations();
  /// _work is used for evaluating a function
  /// f_0 and f_h should be gained from the tail
  real_type* wptr = _work.data() + (_work.span() - 2 * m);
  real_type_1d_view_type f_0(wptr, m);
  wptr += f_0.span();
  real_type_1d_view_type f_h(wptr, m);
  wptr += f_h.span();

  /// use the default values
  real_type fac_min(-1), fac_max(-1);

  Tines::NumericalJacobianForwardDifference<value_type, device_type>::invoke(
        member, *this, fac_min, fac_max, _fac, x, f_0, f_h, J);

  // Tines::NumericalJacobianCentralDifference<real_type, device_type>::invoke(
  //   member, *this, fac_min, fac_max, _fac, x, f_0, f_h, J);
  //
  //   Tines::NumericalJacobianRichardsonExtrapolation<value_type, device_type>::invoke
  //    (member, *this, fac_min, fac_max, _fac, x, f_0, f_h, J);
    member.team_barrier();


}
  //
  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION void computeNumericalJacobianRichardsonExtrapolation(const MemberType& member,
                                              const real_type_1d_view_type& x,
                                              const real_type_2d_view_type& J) const
{
  const ordinal_type m = getNumberOfEquations();
  /// _work is used for evaluating a function
  /// f_0 and f_h should be gained from the tail
  real_type* wptr = _work.data() + (_work.span() - 2 * m);
  real_type_1d_view_type f_0(wptr, m);
  wptr += f_0.span();
  real_type_1d_view_type f_h(wptr, m);
  wptr += f_h.span();

  /// use the default values
  real_type fac_min(-1), fac_max(-1);

  Tines::NumericalJacobianRichardsonExtrapolation<value_type, device_type>::invoke
     (member, *this, fac_min, fac_max, _fac, x, f_0, f_h, J);
  member.team_barrier();


}




  };


}
}

#endif
