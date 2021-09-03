#ifndef __TCHEM_IMPL_ALGEBRAIC_CONSTRAINTS_SURFACE_PROBLEM_HPP__
#define __TCHEM_IMPL_ALGEBRAIC_CONSTRAINTS_SURFACE_PROBLEM_HPP__

#include "Tines_Internal.hpp"

namespace TChem {
namespace Impl {
template <typename ValueType, typename DeviceType>
struct AlgebraicConstraintsSurface_Problem {
  using value_type = ValueType;
  using device_type = DeviceType;
  using scalar_type = typename ats<value_type>::scalar_type;

  using real_type = scalar_type;
  using real_type_1d_view_type = Tines::value_type_1d_view<value_type, device_type>;
  using real_type_2d_view_type = Tines::value_type_2d_view<value_type, device_type>;

  using problem_type =
    TChem::Impl::SimpleSurface_Problem<value_type,
                                       device_type>;

  problem_type _problem;
  real_type_1d_view_type  _unr;


  KOKKOS_INLINE_FUNCTION
int getNumberOfTimeODEs() const { return _problem.getNumberOfTimeODEs(); }

KOKKOS_INLINE_FUNCTION
int getNumberOfConstraints() const {
  return _problem.getNumberOfConstraints();
}

KOKKOS_INLINE_FUNCTION
int getNumberOfEquations() const { return _problem.getNumberOfEquations(); }

template <typename MemberType>
KOKKOS_INLINE_FUNCTION void
computeInitValues(const MemberType &member,
                  const real_type_1d_view_type &u) const {
  const int m = _problem.getNumberOfEquations();
  Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                       [&](const int &i) { u(i) = _unr(i); });
  member.team_barrier();
}

template <typename MemberType>
KOKKOS_INLINE_FUNCTION void
computeFunction(const MemberType &member, const real_type_1d_view_type &u,
                const real_type_1d_view_type &f) const {

  const int m = _problem.getNumberOfEquations();
  _problem.computeFunction(member, u, f);

  member.team_barrier();

  for (ordinal_type k = 0; k < m; k++) {
     f(k) *=  _problem._kmcdSurf.sitedensity*real_type(10.);
  }
  using reducer_type = Tines::SumReducer<value_type>;
  typename reducer_type::value_type usum(0);
  Kokkos::parallel_reduce(
    Kokkos::TeamVectorRange(member, m),
    [&](const ordinal_type& k, typename reducer_type::value_type& update) {
      update += u(k); // Units of omega (kg/m3/s).

    },reducer_type(usum));

  member.team_barrier();
  // const value_type one(1);
  f(m - 1) = real_type(1.) - usum;
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

  const ordinal_type m = _problem.getNumberOfEquations();
  /// _work is used for evaluating a function
  /// f_0 and f_h should be gained from the tail
  real_type* wptr = _problem._work.data() + (_problem._work.span() - 2 * m);
  real_type_1d_view_type f_0(wptr, m);
  wptr += f_0.span();
  real_type_1d_view_type f_h(wptr, m);
  wptr += f_h.span();

  /// use the default values
  const real_type fac_min(-1), fac_max(-1);
  Tines::NumericalJacobianForwardDifference<value_type, device_type>::invoke(
        member, *this, fac_min, fac_max, _problem._fac, x, f_0, f_h, J);
  // NumericalJacobianCentralDifference::team_invoke_detail(
  //   member, *this, fac_min, fac_max, _fac, x, f_0, f_h, J);
  // NumericalJacobianRichardsonExtrapolation::team_invoke_detail
  //  (member, *this, fac_min, fac_max, _fac, x, f_0, f_h, J);

}


};

} // namespace Impl
} // namespace TChem

#endif
