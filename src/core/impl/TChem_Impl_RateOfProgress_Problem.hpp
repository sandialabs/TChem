#ifndef __TCHEM_RATE_OF_PROGRESS_PROBLEM_HPP__
#define __TCHEM_RATE_OF_PROGRESS_PROBLEM_HPP__

#include "Tines_Internal.hpp"
#include "TChem_Impl_CpMixMs.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_Impl_RateOfProgressInd.hpp"
#include "TChem_Impl_MolarWeights.hpp"
#include "TChem_Impl_EnthalpySpecMs.hpp"
#include "TChem_Impl_RateOfProgessJacobian.hpp"

// #define TCHEM_ENABLE_CPMIX
namespace TChem {
namespace Impl {

  template<typename ValueType, typename DeviceType>
  struct RateOfProgess_Problem
  {
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
    real_type_1d_view_type _fac;
    kinetic_model_type _kmcd;
    real_type _p;

    KOKKOS_DEFAULTED_FUNCTION
    RateOfProgess_Problem() = default;

    KOKKOS_INLINE_FUNCTION
    static ordinal_type getNumberOfTimeODEs(const kinetic_model_type& kmcd)
    {
#if defined(TCHEM_ENABLE_CPMIX)
      return  1;
#else
      return /*2*/kmcd.nReac;
#endif
    }

    KOKKOS_INLINE_FUNCTION
    static ordinal_type getNumberOfConstraints(
      const kinetic_model_type& kmcd)
    {
      return 0;
    }

    KOKKOS_INLINE_FUNCTION
    static ordinal_type getNumberOfVariables(
      const kinetic_model_type& kmcd)
    {
#if defined(TCHEM_ENABLE_CPMIX)
      return kmcd.nSpec + 1;
#else
      return kmcd.nSpec + 2;
#endif
    }

    KOKKOS_INLINE_FUNCTION
    static ordinal_type getNumberOfEquations(
      const kinetic_model_type& kmcd)
    {
      return getNumberOfTimeODEs(kmcd) + getNumberOfConstraints(kmcd);
    }


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
    ordinal_type getNumberOfVariables() const
    {
      return getNumberOfVariables(_kmcd);
    }

    KOKKOS_INLINE_FUNCTION
    ordinal_type getNumberOfEquations() const
    {
      return getNumberOfTimeODEs() + getNumberOfConstraints();
    }

    KOKKOS_INLINE_FUNCTION
    ordinal_type getWorkSpaceSize() const { return getWorkSpaceSize(_kmcd); }

    KOKKOS_INLINE_FUNCTION
    static ordinal_type getWorkSpaceSize(const kinetic_model_type& kmcd)
    {
      const ordinal_type m = getNumberOfEquations(kmcd);
      const ordinal_type n = getNumberOfVariables(kmcd);
#if defined(TCHEM_ENABLE_CPMIX)
      const ordinal_type wlen = 2*kmcd.nSpec + m + n;
#else
      const ordinal_type wlen = RateOfProgressInd<real_type, device_type>::getWorkSpaceSize(kmcd)
      + m + n + 2*kmcd.nReac;
#endif
      const ordinal_type workspace_net_production_rate_jac = RateOfProgessJacobian::getWorkSpaceSize(kmcd);
      const ordinal_type workspace_sacado_jac = wlen*ats<value_type>::sacadoStorageCapacity();

      const ordinal_type workspace_size =
        (workspace_net_production_rate_jac > workspace_sacado_jac ? workspace_net_production_rate_jac
                                                 : workspace_sacado_jac);
      return workspace_size;

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
      const scalar_type density = x(0);
      const scalar_type t = x(1);
      const real_type_1d_view_type Ys(&x(2), _kmcd.nSpec);

#if defined(TCHEM_ENABLE_CPMIX)
      auto w = (real_type*)_work.data();
      auto cpks = real_type_1d_view_type(w, _kmcd.nSpec);
      w += _kmcd.nSpec;
      f(0) = CpMixMs<real_type, device_type>::team_invoke(member, t, Ys, cpks, _kmcd);
      member.team_barrier();
#else
      // const real_type Wmix = MolarWeights<real_type, device_type>
      // ::team_invoke(member, Ys, _kmcd);
      // const real_type density = _p * Wmix/ _kmcd.Runiv / t ;
      auto w = (real_type*)_work.data();
      auto ropFor = real_type_1d_view_type(w, _kmcd.nReac);
      w += _kmcd.nReac;
      auto ropRev = real_type_1d_view_type(w, _kmcd.nReac);
      w += _kmcd.nReac;

      RateOfProgressInd<real_type, device_type>::team_invoke(member,
                                                            t,
                                                            _p,
                                                            density,
                                                            Ys,
                                                            ropFor,
                                                            ropRev,
                                                            _work,
                                                            _kmcd);
      //
      member.team_barrier();
      for (size_t k = 0; k < _kmcd.nReac; k++) {
        f(k) = ropFor(k) - ropRev(k) ;
        // f(k+_kmcd.nReac)= ;
      }



#endif
   }

    template<typename MemberType>
    KOKKOS_INLINE_FUNCTION
    void computeFunctionSacado(const MemberType& member,
			       const value_type_1d_view_type& x,
			       const value_type_1d_view_type& f) const
    {
      const value_type density = x(0);
      const value_type t = x(1);
      using range_type = Kokkos::pair<ordinal_type, ordinal_type>;
      const value_type_1d_view_type Ys = Kokkos::subview(x,range_type(2, _kmcd.nSpec + 2  ));

#if defined(TCHEM_ENABLE_CPMIX)
      auto w = (real_type*)_work.data();
      const ordinal_type len = value_type().length();
      const ordinal_type sacadoStorageDimension = ats<value_type>::sacadoStorageDimension(t);
      auto cpks = value_type_1d_view_type(w, _kmcd.nSpec, sacadoStorageDimension);
      w += _kmcd.nSpec*len;
      f(0) = CpMixMs<value_type, device_type>::team_invoke(member, t, Ys, cpks, _kmcd);

#else
    // const value_type Wmix = MolarWeights<value_type, device_type>
    // ::team_invoke(member, Ys, _kmcd);
    // const value_type density = _p * Wmix/ _kmcd.Runiv / t ;

    auto w = (real_type*)_work.data();
    const ordinal_type len = ats<value_type>::sacadoStorageCapacity();
    const ordinal_type sacadoStorageDimension = ats<value_type>::sacadoStorageDimension(t);
    auto ropFor = value_type_1d_view_type(w, _kmcd.nReac, sacadoStorageDimension);
    w += _kmcd.nReac*len;
    auto ropRev = value_type_1d_view_type(w, _kmcd.nReac, sacadoStorageDimension);
    w += _kmcd.nReac*len;

    RateOfProgressInd<value_type, device_type>::team_invoke_sacado(member,
                                                          t,
                                                          _p,
                                                          density,
                                                          Ys,
                                                          ropFor,
                                                          ropRev,
                                                          _work,
                                                          _kmcd);
    member.team_barrier();
    for (size_t k = 0; k < _kmcd.nReac; k++) {
      f(k) = ropFor(k)- ropRev(k);
      // f(k+_kmcd.nReac)= ropRev(k);
    }

#endif
      member.team_barrier();
    }

    /// this one is used in time integration nonlinear solve
    template<typename MemberType>
    KOKKOS_INLINE_FUNCTION
    void computeJacobianSacado(const MemberType& member,
				       const real_type_1d_view_type& s,
				       const real_type_2d_view_type& J) const
    {

      const ordinal_type len = ats<value_type>::sacadoStorageCapacity();
      const ordinal_type m = getNumberOfEquations();
      const ordinal_type n = getNumberOfVariables();

      real_type* wptr = _work.data() + (_work.span() - (m+n)*len  );
      value_type_1d_view_type x(wptr, n, n+1); wptr += n*len;
      value_type_1d_view_type f(wptr, m, n+1); wptr += m*len;

      Kokkos::parallel_for
	(Tines::RangeFactory<value_type>::TeamVectorRange(member, n),
	 [=](const int &i) {
	   x(i) = value_type(n, i, s(i));
	 });
      member.team_barrier();
      computeFunctionSacado(member, x, f);
      member.team_barrier();
      Kokkos::parallel_for
	(Kokkos::TeamThreadRange(member, m),
	 [=](const int &i) {
	   Kokkos::parallel_for
	     (Kokkos::ThreadVectorRange(member, n),
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

  }

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION void computeAnalyticalJacobian(const MemberType& member,
                                              const real_type_1d_view_type& x,
                                              const real_type_2d_view_type& J) const
  {
    const real_type t = x(1);
    const real_type_1d_view_type Ys(&x(2), _kmcd.nSpec);
#if defined(TCHEM_ENABLE_CPMIX)
    auto w = (real_type*)_work.data();
    auto cpks = real_type_1d_view_type(w, _kmcd.nSpec);
    w += _kmcd.nSpec;
    auto hks = real_type_1d_view_type(w, _kmcd.nSpec);
    w += _kmcd.nSpec;

    J(0,0) =
      CpMixMsDerivative::team_invoke(member, t, Ys, cpks, _kmcd);
    //
    member.team_barrier();
    auto CpMix = CpMixMs<real_type, device_type>::team_invoke(member, t, Ys, cpks, _kmcd);
    EnthalpySpecMsFcn<real_type, device_type>::team_invoke(member, t, hks, cpks, _kmcd);

    for (size_t k = 0; k < _kmcd.nSpec; k++) {
      J(0,k+1) =cpks(k);
    }
#else
    Impl::RateOfProgessJacobian::team_invoke(member, t, _p, Ys, J, _work, _kmcd);
#endif
    member.team_barrier();

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
  member.team_barrier();
  // NumericalJacobianCentralDifference::team_invoke_detail(
  //   member, *this, fac_min, fac_max, _fac, x, f_0, f_h, J);
  // NumericalJacobianRichardsonExtrapolation::team_invoke_detail
  //  (member, *this, fac_min, fac_max, _fac, x, f_0, f_h, J);

}




  };


}
}

#endif
