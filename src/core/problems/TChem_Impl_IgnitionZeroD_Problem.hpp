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

#include "Tines_Internal.hpp"

#include "TChem_KineticModelData.hpp"
#include "TChem_Impl_SourceTerm.hpp"
#include "TChem_Impl_JacobianReduced.hpp"

namespace TChem {
  namespace Impl {

    template<typename ValueType, typename DeviceType>
    struct IgnitionZeroD_Problem {
      using value_type = ValueType;
      using device_type = DeviceType;
      using scalar_type = typename ats<value_type>::scalar_type;

      using real_type = scalar_type;
      using real_type_0d_view_type = Tines::value_type_0d_view<real_type,device_type>;
      using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;
      using real_type_2d_view_type = Tines::value_type_2d_view<real_type,device_type>;

      /// sacado is value type
      using value_type_0d_view_type = Tines::value_type_0d_view<value_type,device_type>;
      using value_type_1d_view_type = Tines::value_type_1d_view<value_type,device_type>;
      using value_type_2d_view_type = Tines::value_type_2d_view<value_type,device_type>;
      using kinetic_model_type= KineticModelConstData<device_type>;

      real_type_1d_view_type _work;
      real_type_1d_view_type _work_cvode;
      real_type_1d_view_type _fac;
      kinetic_model_type _kmcd;
      real_type _p;

      KOKKOS_DEFAULTED_FUNCTION
      IgnitionZeroD_Problem() = default;

      KOKKOS_INLINE_FUNCTION
      static ordinal_type getNumberOfTimeODEs(const kinetic_model_type& kmcd) {
	return kmcd.nSpec + 1;
      }

      KOKKOS_INLINE_FUNCTION
      static ordinal_type getNumberOfConstraints(const kinetic_model_type& kmcd) {
	return 0;
      }

      KOKKOS_INLINE_FUNCTION
      static ordinal_type getNumberOfEquations(const kinetic_model_type& kmcd) {
	return getNumberOfTimeODEs(kmcd) + getNumberOfConstraints(kmcd);
      }


      KOKKOS_INLINE_FUNCTION
      ordinal_type getNumberOfTimeODEs() const {
	return getNumberOfTimeODEs(_kmcd);
      }

      KOKKOS_INLINE_FUNCTION
      ordinal_type getNumberOfConstraints() const {
	return getNumberOfConstraints(_kmcd);
      }

      KOKKOS_INLINE_FUNCTION
      ordinal_type getNumberOfEquations() const {
	return getNumberOfTimeODEs() + getNumberOfConstraints();
      }

      KOKKOS_INLINE_FUNCTION
      ordinal_type getWorkSpaceSize() const { return getWorkSpaceSize(_kmcd); }

      KOKKOS_INLINE_FUNCTION
      static ordinal_type getWorkSpaceSize(const kinetic_model_type& kmcd) {
	const ordinal_type src_workspace_size = SourceTerm<value_type, device_type>::getWorkSpaceSize(kmcd);
	const ordinal_type jac_workspace_size = JacobianReduced::getWorkSpaceSize(kmcd);
	const ordinal_type workspace_size_analitical_jacobian =
	  (jac_workspace_size > src_workspace_size ? jac_workspace_size : src_workspace_size);

	const ordinal_type m = getNumberOfEquations(kmcd);
	const ordinal_type workspace_size_sacado_numerical_jacobian =
	  src_workspace_size + 2 * m * ats<value_type>::sacadoStorageCapacity();

	const ordinal_type workspace_size =
	  (workspace_size_analitical_jacobian > workspace_size_sacado_numerical_jacobian
	   ? workspace_size_analitical_jacobian : workspace_size_sacado_numerical_jacobian);

	return workspace_size;
      }

      KOKKOS_INLINE_FUNCTION
      void setWorkspace(const real_type_1d_view_type& work) {
	_work = work;
      }

      /// mxm matrix storage
      KOKKOS_INLINE_FUNCTION
      void setWorkspaceCVODE(const real_type_1d_view_type& work_cvode) {
	_work_cvode = work_cvode;
      }

      template<typename MemberType>
      KOKKOS_INLINE_FUNCTION void computeFunction(const MemberType& member,
						  const real_type_1d_view_type& x,
						  const real_type_1d_view_type& f) const {
	const scalar_type t = x(0);
	const real_type_1d_view_type Ys(&x(1), _kmcd.nSpec);
	Impl::SourceTerm<real_type, device_type>::team_invoke(member, t, _p, Ys, f, _work, _kmcd);
	member.team_barrier();
      }

      template<typename MemberType>
      KOKKOS_INLINE_FUNCTION
      void computeFunctionSacado(const MemberType& member,
				 const value_type_1d_view_type& x,
				 const value_type_1d_view_type& f) const {
	const value_type t = x(0);
	using range_type = Kokkos::pair<ordinal_type, ordinal_type>;
	const value_type_1d_view_type Ys = Kokkos::subview(x,range_type(1, _kmcd.nSpec + 1  ));

	SourceTerm<value_type, device_type>::
	  team_invoke_sacado(member, t, _p, Ys, f, //omega
			     _work, _kmcd);

	member.team_barrier();
      }

      /// this one is used in time integration nonlinear solve
      template<typename MemberType>
      KOKKOS_INLINE_FUNCTION
      void computeSacadoJacobian(const MemberType& member,
				 const real_type_1d_view_type& s,
				 const real_type_2d_view_type& J) const {
#if TCHEM_ETI_FAD_SIZE > 0
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

#else
	Kokkos::abort("Error: Sacado cannot be used with scalar instanciation");
#endif
      }

      //
      /// this one is used in time integration nonlinear solve
      template<typename MemberType>
      KOKKOS_INLINE_FUNCTION
      void computeJacobian(const MemberType& member,
			   const real_type_1d_view_type& s,
			   const real_type_2d_view_type& J) const {
#if defined(TCHEM_ENABLE_SACADO_JACOBIAN_IGNITION_ZERO_D_REACTOR)
	computeSacadoJacobian(member, s, J);
#elif defined(TCHEM_ENABLE_NUMERICAL_JACOBIAN_IGNITION_ZERO_D_REACTOR)
	computeNumericalJacobian(member, s, J);
#else
	computeAnalyticalJacobian(member, s, J);
#endif
      }

      template<typename MemberType>
      KOKKOS_INLINE_FUNCTION void computeAnalyticalJacobian(const MemberType& member,
							    const real_type_1d_view_type& x,
							    const real_type_2d_view_type& J) const {
	const real_type t = x(0);
	const real_type_1d_view_type Ys(&x(1), _kmcd.nSpec);
	Impl::JacobianReduced::team_invoke(member, t, _p, Ys, J, _work, _kmcd);
	member.team_barrier();
      }

      template<typename MemberType>
      KOKKOS_INLINE_FUNCTION void computeNumericalJacobian(const MemberType& member,
							   const real_type_1d_view_type& x,
							   const real_type_2d_view_type& J) const {
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

	Tines::NumericalJacobianForwardDifference<value_type, device_type>
	  ::invoke(member, *this, fac_min, fac_max, _fac, x, f_0, f_h, J);
	member.team_barrier();
	// NumericalJacobianCentralDifference::team_invoke_detail(
	//   member, *this, fac_min, fac_max, _fac, x, f_0, f_h, J);
	// NumericalJacobianRichardsonExtrapolation::team_invoke_detail
	//  (member, *this, fac_min, fac_max, _fac, x, f_0, f_h, J);

      }

      //
      template<typename MemberType>
      KOKKOS_INLINE_FUNCTION void computeNumericalJacobianRichardsonExtrapolation(const MemberType& member,
										  const real_type_1d_view_type& x,
										  const real_type_2d_view_type& J) const  {
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

#if defined(TINES_ENABLE_TPL_SUNDIALS)
#include "Tines_Interface.hpp"

namespace TChem {
  namespace Impl {

    static int ProblemIgnitionZeroD_ComputeFunctionCVODE(realtype t,
							 N_Vector u,
							 N_Vector f,
							 void *user_data) {
      using host_device_type = Tines::UseThisDevice<Kokkos::Serial>::type;
      using problem_type = IgnitionZeroD_Problem<realtype,host_device_type>;
      using realtype_1d_view_type = Tines::value_type_1d_view<realtype, host_device_type>;

      problem_type * problem = (problem_type*)(user_data);
      TINES_CHECK_ERROR(problem == nullptr, "user data is failed to cast to problem type");

      int m = problem->getNumberOfEquations();
      const auto member = Tines::HostSerialTeamMember();

      realtype * u_data = N_VGetArrayPointer_Serial(u);
      realtype * f_data = N_VGetArrayPointer_Serial(f);

      realtype_1d_view_type uu(u_data, m);
      realtype_1d_view_type ff(f_data, m);

      problem->computeFunction(member, uu, ff);
      return 0;
    }

    static int ProblemIgnitionZeroD_ComputeJacobianCVODE(realtype t,
							 N_Vector u,
							 N_Vector f,
							 SUNMatrix J,
							 void *user_data,
							 N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
      using host_device_type = Tines::UseThisDevice<Kokkos::Serial>::type;
      using problem_type = IgnitionZeroD_Problem<realtype,host_device_type>;
      using realtype_1d_view_type = Tines::value_type_1d_view<realtype, host_device_type>;
      using realtype_2d_view_type = Tines::value_type_2d_view<realtype, host_device_type>;

      problem_type * problem = (problem_type*)(user_data);;
      TINES_CHECK_ERROR(problem == nullptr, "user data is failed to cast to problem type");

      int m = problem->getNumberOfEquations();
      const auto member = Tines::HostSerialTeamMember();

      realtype * u_data = N_VGetArrayPointer_Serial(u);

      realtype_1d_view_type uu(u_data, m);
      realtype_2d_view_type JJ(problem->_work_cvode.data(), m, m);

#if defined(TCHEM_ENABLE_SACADO_JACOBIAN_IGNITION_ZERO_D_REACTOR)
      Kokkos::abort("Error: Sacado cannot be used with CVODE");
#elif defined(TCHEM_ENABLE_NUMERICAL_JACOBIAN_IGNITION_ZERO_D_REACTOR)
      problem->computeNumericalJacobian(member, uu, JJ);
#else
      problem->computeAnalyticalJacobian(member, uu, JJ);
#endif

      /// prevent potential mismatch of data layout between kokkos and sundials
      for (int i=0;i<m;++i)
	for (int j=0;j<m;++j)
	  SM_ELEMENT_D(J, i, j) = JJ(i,j);
      return 0;
    }

  }
}

#endif
#endif
