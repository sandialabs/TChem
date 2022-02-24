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
#ifndef __TCHEM_IMPL_ISOTHERMAL_TRANSIENT_CONT_STIRRED_TANK_REACTOR_PROBLEM_HPP__
#define __TCHEM_IMPL_ISOTHERMAL_TRANSIENT_CONT_STIRRED_TANK_REACTOR_PROBLEM_HPP__

/* This problem advance a cstr with surface reactions.
 */
#include "Tines_Internal.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_Impl_MolarWeights.hpp"
#include "TChem_Impl_IsothermalTransientContStirredTankReactorRHS.hpp"
#include "TChem_Util.hpp"

namespace TChem {
namespace Impl {

  template<typename ValueType, typename DeviceType>
struct IsothermalTransientContStirredTankReactor_Problem
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
  using cstr_data_type = TransientContStirredTankReactorData<device_type>;


  KOKKOS_DEFAULTED_FUNCTION
  IsothermalTransientContStirredTankReactor_Problem() = default;
  /// public access to these member functions
  real_type_1d_view_type _x;
  real_type_1d_view_type _work;
  real_type_1d_view_type _fac; /// numerical jacobian
  kinetic_model_type _kmcd;
  kinetic_surf_model_type _kmcdSurf;
  cstr_data_type _cstr;
  real_type _temperature;
  real_type_0d_view_type _mass_flow_out; // outlet mass flow rate

  KOKKOS_INLINE_FUNCTION
  static ordinal_type getNumberOfTimeODEs(const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf, const cstr_data_type& cstr)
  {
    return kmcd.nSpec + kmcdSurf.nSpec - cstr.number_of_algebraic_constraints;
  }

  KOKKOS_INLINE_FUNCTION
  static ordinal_type getNumberOfConstraints(
    const cstr_data_type& cstr)
  {
    return cstr.number_of_algebraic_constraints; //
  }

  KOKKOS_INLINE_FUNCTION
  static ordinal_type getNumberOfEquations(
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf)
  {
    return kmcd.nSpec + kmcdSurf.nSpec;
  }

  ///
  /// non static functions that require the object
  /// workspace size is required without kmcd
  ///

  KOKKOS_INLINE_FUNCTION
  ordinal_type getNumberOfEquations() const
  {
    return getNumberOfTimeODEs() + getNumberOfConstraints();
  }

  KOKKOS_INLINE_FUNCTION
  ordinal_type getNumberOfTimeODEs() const
  {
    return getNumberOfTimeODEs(_kmcd, _kmcdSurf, _cstr);
  }

  KOKKOS_INLINE_FUNCTION
  ordinal_type getNumberOfConstraints() const
  {
    return getNumberOfConstraints(_cstr);
  }


  KOKKOS_INLINE_FUNCTION
  static ordinal_type getWorkSpaceSize(
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf)
  {
    const ordinal_type jac_workspace_size = 2 * getNumberOfEquations(kmcd, kmcdSurf);
    const ordinal_type src_workspace_size = IsothermalTransientContStirredTankReactorRHS<value_type, device_type>
    ::getWorkSpaceSize(kmcd, kmcdSurf);//
    const ordinal_type workspace_size = src_workspace_size + jac_workspace_size;

    return workspace_size*ats<value_type>::sacadoStorageCapacity();
  }

  KOKKOS_INLINE_FUNCTION
  ordinal_type getWorkSpaceSize() const
  {
    return getWorkSpaceSize(_kmcd, _kmcdSurf);
  }

  KOKKOS_INLINE_FUNCTION
  void setWorkspace(const real_type_1d_view_type& work)
  {
    _work = work;
  }

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION
  void computeFunctionSacado(const MemberType& member,
           const value_type_1d_view_type& x,
           const value_type_1d_view_type& f) const
  {
    using range_type = Kokkos::pair<ordinal_type, ordinal_type>;
    const value_type_1d_view_type Ys = Kokkos::subview(x,range_type(0, _kmcd.nSpec  ));
    const value_type Wmix = MolarWeights<value_type, device_type>
    ::team_invoke(member, Ys, _kmcd);
    const value_type density = _cstr.pressure * Wmix/ _kmcd.Runiv / _temperature ;

    const value_type_1d_view_type siteFraction =
    Kokkos::subview(x,range_type(_kmcd.nSpec, _kmcd.nSpec +_kmcdSurf.nSpec  ));

    IsothermalTransientContStirredTankReactorRHS<value_type, device_type>::team_invoke_sacado(member,
                                           _temperature, // constant temperature
                                           Ys,
                                           siteFraction,
                                           density,
                                           _cstr.pressure, // constant pressure
                                           f,
                                           _mass_flow_out, // outlet mass flow rate
                                           _work,
                                           _kmcd,
                                           _kmcdSurf,
                                           _cstr);

    member.team_barrier();


#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      std::cout << "temperature " << _temperature << '\n';
      std::cout << "pressure " << _cstr.pressure << '\n';

      for (ordinal_type i = 0; i < _kmcd.nSpec; i++)
        std::cout << "Ys(i) "<< i << " "<< Ys(i) << '\n';

      for (ordinal_type i = 0; i < _kmcdSurf.nSpec; i++)
      std::cout << "siteFraction(i) "<< i << " " << siteFraction(i) << '\n';

      for (ordinal_type i = 0; i < _kmcdSurf.nSpec + _kmcd.nSpec; i++)
       std::cout << "f(i) "<< i << " " << f(i) << '\n';
    }
#endif

  }


  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION void computeFunction(const MemberType& member,
                                                const real_type_1d_view_type& x,
                                                const real_type_1d_view_type& f) const
  {
    const real_type_1d_view_type Ys(&x(0), _kmcd.nSpec);
    const real_type_1d_view_type siteFraction(&x(_kmcd.nSpec), _kmcdSurf.nSpec);
      //compute density
    const real_type Wmix = MolarWeights<real_type, device_type>::team_invoke(member, Ys, _kmcd);
      // compute pressure
    const real_type density = _cstr.pressure * Wmix/ _kmcd.Runiv / _temperature ;

    IsothermalTransientContStirredTankReactorRHS<real_type, device_type>::team_invoke(member,
                                             _temperature, // constant temperature
                                             Ys,
                                             siteFraction,
                                             density,
                                             _cstr.pressure, // constant pressure
                                             f,
                                             _mass_flow_out, // outlet mass flow rate
                                             _work,
                                             _kmcd,
                                             _kmcdSurf,
                                             _cstr);

#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      std::cout << "temperature " << _temperature << '\n';
      std::cout << "pressure " << _cstr.pressure << '\n';

      for (ordinal_type i = 0; i < _kmcd.nSpec; i++)
          std::cout << "Ys(i) "<< i << " "<< Ys(i) << '\n';

      for (ordinal_type i = 0; i < _kmcdSurf.nSpec; i++)
          std::cout << "siteFraction(i) "<< i << " " << siteFraction(i) << '\n';

      for (ordinal_type i = 0; i < _kmcdSurf.nSpec + _kmcd.nSpec; i++)
          std::cout << "f(i) "<< i << " " << f(i) << '\n';

      FILE* fs = fopen(
          "IsothermalTransientContStirredTankReactor_Problem_computeFunction.team_invoke.test.out", "a+");
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
    // printf("{ \"Jacobian\":[");
    // for (ordinal_type i = 0; i < m; i++) {
    //   for (ordinal_type j = 0; j < m; j++) {
    //     printf("%e, ", J(i,j) );
    //   }
    // }
    // printf("]}\n");


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

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION
  void computeJacobian(const MemberType& member,
             const real_type_1d_view_type& s,
             const real_type_2d_view_type& J) const
  {
#if defined(TCHEM_ENABLE_SACADO_JACOBIAN_TRANSIENT_CONT_STIRRED_TANK_REACTOR)
     computeSacadoJacobian(member, s, J);
#else
     computeNumericalJacobian(member, s, J);
#endif
  }


};

} // namespace Impl
} // namespace TChem

#endif
