#ifndef __TCHEM_IMPL_TANGENT_LINEAR_APPROXIMATION_IGNITION_DELAY_TIME_HPP__
#define __TCHEM_IMPL_TANGENT_LINEAR_APPROXIMATION_IGNITION_DELAY_TIME_HPP__

#include "TChem_Util.hpp"

namespace TChem {
  namespace Impl {
    
    template<typename ValueType, typename DeviceType,
	     template<typename,typename> class SourceTermType>
    struct TangentLinearApproximationIgnitionDelayTime {
      using value_type = ValueType;
      using device_type = DeviceType;
      
      static_assert(!ats<value_type>::is_sacado, "Error: SACADO type cannot be used");

      using value_type_0d_view_type = Tines::value_type_0d_view<value_type,device_type>;
      using value_type_1d_view_type = Tines::value_type_1d_view<value_type,device_type>;
      using value_type_2d_view_type = Tines::value_type_2d_view<value_type,device_type>;
      using kinetic_model_type= KineticModelConstData<device_type>;

      KOKKOS_INLINE_FUNCTION
      static ordinal_type getWorkSpaceSize(const kinetic_model_type& kmcd) {
	using tla_source_term_type = SourceTermType<value_type,device_type>;
	
	const ordinal_type
	  m = tla_source_term_type::getNumberOfEquations(kmcd),
	  n = tla_source_term_type::getNumberOfParameters(kmcd);

	const ordinal_type wsize_source_term = tla_source_term_type::getWorkSpaceSize(kmcd);
	ordinal_type wsize_solve(0);
	{
	  value_type_2d_view_type A(nullptr, m, m), B(nullptr, m, n);
	  Tines::SolveLinearSystem::workspace(A, B, wsize_solve);
	}
	/// work_source_term
	const ordinal_type wsize_a =  wsize_source_term;
	/// S_a + S_b + A + work_solve
	const ordinal_type wsize_b = 2*m*n + m*m + wsize_solve;
	
	return wsize_a > wsize_b ? wsize_a : wsize_b;
      }
      
      template<typename MemberType>
      KOKKOS_INLINE_FUNCTION
      static void
      team_invoke(const MemberType& member,
		  /// time integration method
		  const value_type& theta,
		  /// input delta t
		  const value_type& dt,
		  /// input state vector of reactor and Jacobian
		  const value_type& density,
		  const value_type& pressure_a, const value_type& pressure_b,
		  const value_type& temperature_a, const value_type& temperature_b,
		  const value_type_1d_view_type& Ys_a, const value_type_1d_view_type& Ys_b,
		  const value_type_2d_view_type& J_a, const value_type_2d_view_type& J_b,
		  /// input state vector for the sensitivity
		  const value_type_2d_view_type& state_z, /// mass fraction (nSpec+1, nReac)
		  /// output
		  const value_type_2d_view_type& state_z_out,
		  /// workspace
		  const value_type_1d_view_type& work,
		  /// const input from kinetic model
		  const kinetic_model_type& kmcd) {
	/// problem workspace
	auto wptr = work.data();
	
	using tla_source_term_type = SourceTermType<value_type,device_type>;

	/// compute source term
	const ordinal_type
	  m = tla_source_term_type::getNumberOfEquations(kmcd),
	  n = tla_source_term_type::getNumberOfParameters(kmcd);

	value_type_2d_view_type S_a(wptr, m, n); wptr += S_a.span();
	value_type_2d_view_type S_b(wptr, m, n); wptr += S_b.span();
	{
	  const value_type zero(0), one(1);
	  
	  auto wtmp = wptr;
	  const ordinal_type wsize_source_term = tla_source_term_type::getWorkSpaceSize(kmcd);
	  value_type_1d_view_type work_source_term(wtmp, wsize_source_term); wtmp += work_source_term.span();
	  
	  // if euler does not need S_a
	  if (theta > zero ) {
	    tla_source_term_type
	      ::team_invoke(member,
			    density, pressure_a, temperature_a, Ys_a,
			    S_a,
			    work_source_term,
			    kmcd);
	    member.team_barrier();
	  }

	  if (theta < one) {
	    tla_source_term_type
	      ::team_invoke(member,
			    density, pressure_b, temperature_b, Ys_b,
			    S_b,
			    work_source_term,
			    kmcd);
	    member.team_barrier();
	  }
	}
	
	/// integrate over dt
	/// 0 - backward euler, 0.5 - crank-nicholson, 1 - forward euler
	const value_type one_minus_theta(1.0-theta);
	
	/// compute right hand side
	/// Zn + dt theta (J Zn + Sn)
	const auto S = S_a; 
	{
	  Kokkos::parallel_for
	    (Kokkos::TeamVectorRange(member, m*n),
	     [=](const ordinal_type ij) {
	       const ordinal_type i = ij/n, j = ij%n;
	       value_type val(0);
	       if (theta > 0 ) {
		 for (ordinal_type k=0;k<m;++k)
		   val += J_a(i,k)*state_z(k,j);
	       }
	       S(i,j) = state_z(i,j) + dt*theta*(val + S_a(i,j)) + dt*one_minus_theta*S_b(i,j);
	     });
	  member.team_barrier();
	}

	/// modify the Jacobian
	value_type_2d_view_type A(wptr, m, m); wptr +=  A.span();
	{
	  Kokkos::parallel_for
	    (Kokkos::TeamVectorRange(member, m*m),
	     [=](const ordinal_type ij) {
	       const ordinal_type i = ij/m, j = ij%m;
	       A(i,j) = value_type(i==j) - dt*one_minus_theta*J_b(i,j);
	     });
	  member.team_barrier();
	}

	/// solve the system
	{
	  if (theta < 1) {
	    ordinal_type matrix_rank(0), wlen(0);
	    Tines::SolveLinearSystem::workspace(A, S, wlen);
	    
	    value_type_1d_view_type work_solve(wptr, wlen);
	    Tines::SolveLinearSystem::invoke(member, A, state_z_out, S, work_solve, matrix_rank);
	  } else {
	    Kokkos::parallel_for
	      (Kokkos::TeamVectorRange(member, m*n),
	       [=](const ordinal_type ij) {
		 const ordinal_type i = ij/n, j = ij%n;
		 state_z_out(i,j) = S(i,j);
	       });	    
	  }
	}
	member.team_barrier();	
      }
    };

  } // namespace Impl
} // namespace TChem

#endif
