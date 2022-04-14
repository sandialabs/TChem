#ifndef __TCHEM_IMPL_CONSTANT_VOLUME_IGNITION_REACTOR_TANGENT_LINEAR_APPROXIMATION_SOURCE_TERM_HPP__
#define __TCHEM_IMPL_CONSTANT_VOLUME_IGNITION_REACTOR_TANGENT_LINEAR_APPROXIMATION_SOURCE_TERM_HPP__

#include "TChem_Util.hpp"
#include "TChem_Impl_EnthalpySpecMs.hpp"
#include "TChem_Impl_RateOfProgressInd.hpp"
#include "TChem_Impl_CpMixMs.hpp"
#include "TChem_Impl_MolarWeights.hpp"

namespace TChem {
  namespace Impl {

    /// this does not need to have template value type
    template<typename ValueType, typename DeviceType>
    struct ConstantVolumeIgnitionReactorTangetLinearApproximationSourceTerm {
      using value_type = ValueType;
      using device_type = DeviceType;

      static_assert(!Tines::ats<value_type>::is_sacado, "SACADO type should not be used");

      using value_type_1d_view_type = Tines::value_type_1d_view<value_type,device_type>;
      using value_type_2d_view_type = Tines::value_type_2d_view<value_type,device_type>;

      using kinetic_model_type= KineticModelConstData<device_type>;

      KOKKOS_INLINE_FUNCTION
      static ordinal_type getNumberOfParameters(const kinetic_model_type& kmcd) {
        return kmcd.nReac;
      }

      KOKKOS_INLINE_FUNCTION
      static ordinal_type getNumberOfEquations(const kinetic_model_type& kmcd) {
        return kmcd.nSpec + 1;
      }

      KOKKOS_INLINE_FUNCTION
      static ordinal_type getWorkSpaceSize(const kinetic_model_type& kmcd) {
        const ordinal_type
	  m = getNumberOfEquations(kmcd),
	  n = getNumberOfParameters(kmcd);
	
        const ordinal_type work_len_rop = Impl::RateOfProgressInd<value_type, device_type>::getWorkSpaceSize(kmcd);
	/// rop work space, kfor and krev, S_rev
	return  work_len_rop + kmcd.nReac*2 + m*n;
      }

      template<typename MemberType>
      KOKKOS_INLINE_FUNCTION
      static void
      team_invoke_detail(const MemberType& member,
			 /// input
			 const value_type& density,
			 const value_type& pressure,
			 const value_type& temperature,
			 const value_type_1d_view_type& Ys,
			 /// output
			 const value_type_2d_view_type& S_fwd,
			 const value_type_2d_view_type& S_rev,
			 /// workspace
			 const value_type_1d_view_type& work,
			 const kinetic_model_type& kmcd) {
	const value_type zero(0), one(1);

	using EnthalpySpecMs = EnthalpySpecMsFcn<value_type,device_type>;
	using CpMixMs = Impl::CpMixMs<value_type,device_type>;

	auto wptr = work.data();

	/// 1. compute rate of progress
	value_type_1d_view_type ropFor(wptr, kmcd.nReac); wptr += ropFor.span();
	value_type_1d_view_type ropRev(wptr, kmcd.nReac); wptr += ropRev.span();
	{
	  const ordinal_type wsize_rop = Impl::RateOfProgressInd<value_type,device_type>::getWorkSpaceSize(kmcd);
	  value_type_1d_view_type work_rop(wptr, wsize_rop);
	  RateOfProgressInd<value_type,device_type>
	    ::team_invoke(member,
			  temperature,
			  pressure,
			  density,
			  Ys,
			  ropFor,
			  ropRev,
			  work_rop,
			  kmcd);
	}

	value_type_1d_view_type hks(wptr, kmcd.nSpec); wptr += hks.span();
	value_type_1d_view_type cpks(wptr, kmcd.nSpec); wptr += cpks.span();

	/// 2. compute  cpmix
	const value_type cpmix = CpMixMs::team_invoke(member, temperature, Ys, cpks, kmcd);
	const value_type Wmix = MolarWeights<value_type, device_type>::team_invoke(member, Ys, kmcd);
	member.team_barrier();

	// 3. compute enthalpy
	EnthalpySpecMs::team_invoke(member, temperature, hks, cpks, kmcd);

	// 4. gamma
	const value_type cvmix = cpmix - kmcd.Runiv / Wmix;
	const value_type gamma = cpmix / cvmix;

	/// 5. ensamble S matrices	
        const ordinal_type
	  m = getNumberOfEquations(kmcd),
	  n = getNumberOfParameters(kmcd);

	Kokkos::parallel_for
	  (Kokkos::TeamVectorRange(member, m*n),
	   [=](const ordinal_type& ij) {
	     const ordinal_type i = ij/n, j = ij%n;
	     S_fwd(i,j) = zero;
	     S_rev(i,j) = zero;
	   });
	member.team_barrier();

	Kokkos::parallel_for
	  (Kokkos::TeamVectorRange(member, n), [&](const ordinal_type& i) {
	    const value_type rop_fwd_at_i = ropFor(i)/density;
	    const value_type rop_rev_at_i = ropRev(i)/density;

	    // species equations
	    // reactants
	    value_type sum_term1_fwd(0);
	    value_type sum_term2_fwd(0);
	    value_type sum_term1_rev(0);
	    value_type sum_term2_rev(0);

	    for (ordinal_type j=0;j<kmcd.reacNreac(i);++j) {
	      const ordinal_type kspec = kmcd.reacSidx(i, j);
	      const value_type val_fwd = kmcd.reacNuki(i, j) * rop_fwd_at_i * kmcd.sMass(kspec);
	      const value_type val_rev = -kmcd.reacNuki(i, j) * rop_rev_at_i * kmcd.sMass(kspec);
	      
	      S_fwd(kspec+1 ,i) = val_fwd;
	      S_rev(kspec+1 ,i) = val_rev;
	      
	      sum_term1_fwd += val_fwd * hks(kspec);
	      sum_term2_fwd += val_fwd / kmcd.sMass(kspec);
	      
	      sum_term1_rev += val_rev * hks(kspec);
	      sum_term2_rev += val_rev / kmcd.sMass(kspec);
	    }
	    
	    // productos
	    const ordinal_type joff = kmcd.reacSidx.extent(1) / 2;
	    for (ordinal_type j=0;j<kmcd.reacNprod(i);++j) {
	      const ordinal_type kspec = kmcd.reacSidx(i, j + joff);
	      
	      const value_type val_fwd = S_fwd(kspec+1,i) + kmcd.reacNuki(i, j + joff) * rop_fwd_at_i * kmcd.sMass(kspec);
	      const value_type val_rev = S_rev(kspec+1,i) - kmcd.reacNuki(i, j + joff) * rop_rev_at_i * kmcd.sMass(kspec);
	      
	      S_fwd(kspec+1 ,i) = val_fwd;
	      S_rev(kspec+1 ,i) = val_rev;
	      
	      sum_term1_fwd += val_fwd * hks(kspec);
	      sum_term2_fwd += val_fwd / kmcd.sMass(kspec);
	      
	      sum_term1_rev += val_rev * hks(kspec);
	      sum_term2_rev += val_rev / kmcd.sMass(kspec);
	    }

	    // temperature
	    S_fwd(0,i) = - gamma * sum_term1_fwd / cpmix + (gamma - one) * temperature * Wmix *sum_term2_fwd;
	    S_rev(0,i) = - gamma * sum_term1_rev / cpmix + (gamma - one) * temperature * Wmix *sum_term2_rev;
	  });
      }

      template<typename MemberType>
      KOKKOS_INLINE_FUNCTION
      static void
      team_invoke(const MemberType& member,
		  /// input
		  const value_type& density,
		  const value_type& pressure,
		  const value_type& temperature,
		  const value_type_1d_view_type& Ys,
		  /// output
		  const value_type_2d_view_type& S,
		  /// workspace
		  const value_type_1d_view_type& work,
		  const kinetic_model_type& kmcd)
      {

	const ordinal_type
	  wsize_required = getWorkSpaceSize(kmcd),
	  wsize_provided = work.extent(0);
	TCHEM_CHECK_ERROR(wsize_required > wsize_provided, "Error: workspace provided is smallerthan required");
	
	auto wptr = work.data();
        const ordinal_type
	  m = getNumberOfEquations(kmcd),
	  n = getNumberOfParameters(kmcd);
	const auto S_fwd = S;
      	value_type_2d_view_type S_rev(wptr, m, n); wptr += S_rev.span();

        const ordinal_type wsize_used(wptr - work.data());
        value_type_1d_view_type ww(wptr, wsize_provided - wsize_used);

        team_invoke_detail(member,
                           density,
			   pressure,
                           temperature,
			   Ys,
			   /// output
			   S_fwd,
			   S_rev,
			   /// workspace
			   ww,
			   kmcd);
        member.team_barrier();

        Kokkos::parallel_for
	  (Kokkos::TeamVectorRange(member, m*n),
	   [=](const ordinal_type ij) {
	     const ordinal_type i = ij/n, j = ij%n;
	     S(i,j) = S_rev(i,j) + S_fwd(i,j);
	   });
	member.team_barrier();
      }
    };

  } // namespace Impl
} // namespace TChem

#endif
