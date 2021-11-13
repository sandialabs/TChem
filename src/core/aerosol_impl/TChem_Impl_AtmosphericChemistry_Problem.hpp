#ifndef __TCHEM_ATMOSPHERIC_CHEMISTRY_PROBLEM_HPP__
#define __TCHEM_ATMOSPHERIC_CHEMISTRY_PROBLEM_HPP__

#include "Tines_Internal.hpp"

#include "TChem_KineticModelData.hpp"
#include "TChem_Impl_ReactionRatesAerosol.hpp"

namespace TChem {
namespace Impl {

  template<typename ValueType, typename DeviceType>
  struct AtmosphericChemistry_Problem
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
    using kinetic_model_type= KineticModelNCAR_ConstData<device_type>;

    real_type_1d_view_type _work;
    real_type_1d_view_type _fac;
    // real_type_1d_view_type _omega;
    real_type_1d_view_type _const_concentration;
    kinetic_model_type _kmcd;
    real_type _temperature;
    real_type _pressure;

    KOKKOS_DEFAULTED_FUNCTION
    AtmosphericChemistry_Problem() = default;

    KOKKOS_INLINE_FUNCTION
    static ordinal_type getNumberOfTimeODEs(const kinetic_model_type& kmcd)
    {
      return kmcd.nSpec - kmcd.nConstSpec; // molar concentration
    }

    KOKKOS_INLINE_FUNCTION
    static ordinal_type getNumberOfConstraints(
      const kinetic_model_type& kmcd)
    {
      return 0;
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
    ordinal_type getNumberOfEquations() const
    {
      return getNumberOfTimeODEs() + getNumberOfConstraints();
    }

    KOKKOS_INLINE_FUNCTION
    ordinal_type getWorkSpaceSize() const { return getWorkSpaceSize(_kmcd); }

    KOKKOS_INLINE_FUNCTION
    static ordinal_type getWorkSpaceSize(const kinetic_model_type& kmcd)
    {

      const ordinal_type src_workspace_size
      = ReactionRatesAerosol<value_type, device_type>::getWorkSpaceSize(kmcd);
      const ordinal_type m = getNumberOfEquations(kmcd);

      const ordinal_type workspace_size =
      (src_workspace_size + 2*m)*ats<value_type>::sacadoStorageCapacity();

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

      if (_kmcd.nConstSpec > 0 ) {
        Impl::ReactionRatesAerosol<real_type, device_type>
        ::team_invoke(member, _temperature, _pressure, x, _const_concentration, f, _work,  _kmcd);

      } else {
        Impl::ReactionRatesAerosol<real_type, device_type>
        ::team_invoke(member, _temperature, _pressure, x, f, _work,  _kmcd);
      }


      member.team_barrier();
    }

    template<typename MemberType>
    KOKKOS_INLINE_FUNCTION
    void computeFunctionSacado(const MemberType& member,
			       const value_type_1d_view_type& x,
			       const value_type_1d_view_type& f) const
    {
    if (_kmcd.nConstSpec > 0 ) {
      // Kokkos::abort("Error Atmospheric Chemistry : sacado version does not work with tracer species.\n");
      Impl::ReactionRatesAerosol<value_type, device_type>
      ::team_invoke_sacado(member, _temperature, _pressure, x, _const_concentration, f, _work,  _kmcd);
    } else {
      Impl::ReactionRatesAerosol<value_type, device_type>
          ::team_invoke_sacado(member, _temperature, _pressure, x, f, _work,  _kmcd);
    }

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

#if defined(TCHEM_ENABLE_SACADO_JACOBIAN_ATMOSPHERIC_CHEMISTRY)
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