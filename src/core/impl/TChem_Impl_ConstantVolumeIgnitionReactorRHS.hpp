#ifndef __TCHEM_IMPL_CONST_VOLUME_IGNITION_REACTOR_RHS_HPP__
#define __TCHEM_IMPL_CONST_VOLUME_IGNITION_REACTOR_RHS_HPP__


#include "TChem_Impl_EnthalpySpecMs.hpp"
#include "TChem_Impl_MolarConcentrations.hpp"
#include "TChem_Impl_ReactionRates.hpp"
#include "TChem_Impl_CpMixMs.hpp"
#include "TChem_Impl_MolarWeights.hpp"
#include "TChem_Util.hpp"

namespace TChem {
namespace Impl {

template<typename ValueType, typename DeviceType>
struct ConstantVolumeIgnitionReactorRHS
{

  using value_type = ValueType;
  using device_type = DeviceType;
  using scalar_type = typename ats<value_type>::scalar_type;

  using real_type = scalar_type;
  using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;

  using ordinary_type_1d_view_type = Tines::value_type_1d_view<ordinal_type,device_type>;

  /// sacado is value type
  using value_type_1d_view_type = Tines::value_type_1d_view<value_type,device_type>;
  using kinetic_model_type= KineticModelConstData<device_type>;

  KOKKOS_INLINE_FUNCTION static ordinal_type getWorkSpaceSize(
    const kinetic_model_type& kmcd)
  {
    const ordinal_type work_kfor_rev_size =
    Impl::KForwardReverse<value_type,device_type>::getWorkSpaceSize(kmcd);

    const ordinal_type workspace_size = (4 * kmcd.nSpec + 8 * kmcd.nReac + work_kfor_rev_size);
    return workspace_size;
  }

  ///
  ///  \param t  : temperature [K]
  ///  \param Xc : array of \f$N_{spec}\f$ doubles \f$((XC_1,XC_2,...,XC_N)\f$:
  ///              molar concentrations XC \f$[kmol/m^3]\f$
  ///  \return omega : array of \f$N_{spec}\f$ molar reaction rates
  ///  \f$\dot{\omega}_i\f$ \f$\left[kmol/(m^3\cdot s)\right]\f$
  ///
  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    /// input
    const value_type& t,
    const value_type& p,
    const value_type& density,
    const value_type_1d_view_type& Ys, /// (kmcd.nSpec)
    /// output
    const value_type_1d_view_type& omega_t, /// (1)
    const value_type_1d_view_type& omega,   /// (kmcd.nSpec)
    /// workspace
    const value_type_1d_view_type& gk,
    const value_type_1d_view_type& hks,
    const value_type_1d_view_type& cpks,
    const value_type_1d_view_type& concX,
    const value_type_1d_view_type& concM,
    const value_type_1d_view_type& kfor,
    const value_type_1d_view_type& krev,
    const value_type_1d_view_type& ropFor,
    const value_type_1d_view_type& ropRev,
    const value_type_1d_view_type& Crnd,
    const ordinary_type_1d_view_type& iter,
    const value_type_1d_view_type& work_kfor_rev,
    /// const input from kinetic model
    const kinetic_model_type& kmcd)
  {
    const real_type zero(0), one(1);
    using reducer_type = Tines::SumReducer<value_type>;

    using EnthalpySpecMs = EnthalpySpecMsFcn<value_type,device_type>;
    using CpMixMs = Impl::CpMixMs<value_type, device_type>;

    /// 3. compute  cpmix
    const value_type cpmix = CpMixMs::team_invoke(member, t, Ys, cpks, kmcd);
    member.team_barrier();
    /// 1. compute molar reaction rates
    ReactionRates<value_type,device_type>::team_invoke_detail(member,
                                       t,
                                       p,
                                       density,
                                       Ys,
                                       omega,
                                       gk,
                                       hks,
                                       cpks,
                                       concX,
                                       concM,
                                       kfor,
                                       krev,
                                       ropFor,
                                       ropRev,
                                       Crnd,
                                       iter,
                                       work_kfor_rev,
                                       kmcd);

    /// 2. transform molar reaction rates to mass reaction rates
    Kokkos::parallel_for(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& i) { omega(i) *= kmcd.sMass(i); });

    /// 4. compute species enthalies
    EnthalpySpecMs::team_invoke(member, t, hks, cpks, kmcd);

    const value_type Wmix = MolarWeights<value_type, device_type>
    ::team_invoke(member, Ys, kmcd);

    Kokkos::single(Kokkos::PerTeam(member), [&]() { omega_t(0) = zero; });
    member.team_barrier();
    // 5. compute cv and gamma
    const value_type cvmix = cpmix - kmcd.Runiv / Wmix;
    const value_type gamma = cpmix / cvmix;

    /// 5. transform reaction rates to source term (Wi/rho)
    const value_type orho = one / density;
    Kokkos::parallel_for(Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
                         [&](const ordinal_type& i) { omega(i) *= orho; });

    /// 6. compute source term for temperature

    typename reducer_type::value_type sum_wi_Wi_rho(0);// 1/rho * sum wi/Wi

    Kokkos::parallel_reduce(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& i, typename reducer_type::value_type& update) {
        update += omega(i)/kmcd.sMass(i);
        const value_type val = -omega(i) * hks(i);
        Kokkos::atomic_add(&omega_t(0), val);

    },
    reducer_type(sum_wi_Wi_rho));

    member.team_barrier();
    Kokkos::single(Kokkos::PerTeam(member), [&]() {
      omega_t(0) *= gamma / cpmix;
      omega_t(0) += ( gamma - one ) * t * Wmix  * sum_wi_Wi_rho;

     });

     member.team_barrier();

  }

  template<typename MemberType,typename WorkViewType>
  KOKKOS_FORCEINLINE_FUNCTION static void team_invoke_sacado(
    const MemberType& member,
    /// input
    const value_type& t,
    const value_type& p,
    const value_type& density,
    const value_type_1d_view_type& Ys, /// (kmcd.nSpec)
    /// output
    const value_type_1d_view_type& omega, /// (kmcd.nSpec + 1)
    /// workspace
    const WorkViewType& work,
    /// kmcd.nSpec(5) : Xc, concX,gk,hks,cpks,
    /// kmcd.nReac(5) : concM,kfor,krev,rop,Crnd
    /// const input from kinetic model
    const kinetic_model_type& kmcd)
  {

    auto w = (real_type*)work.data();
    const ordinal_type len = value_type().length();
    const ordinal_type sacadoStorageDimension = ats<value_type>::sacadoStorageDimension(t);

    auto gk = value_type_1d_view_type(w, kmcd.nSpec, sacadoStorageDimension);
    w += kmcd.nSpec*len;
    auto hks = value_type_1d_view_type(w, kmcd.nSpec, sacadoStorageDimension);
    w += kmcd.nSpec*len;
    auto cpks = value_type_1d_view_type(w, kmcd.nSpec, sacadoStorageDimension);
    w += kmcd.nSpec*len;
    auto concX = value_type_1d_view_type(w, kmcd.nSpec, sacadoStorageDimension);
    w += kmcd.nSpec*len;

    auto concM = value_type_1d_view_type(w, kmcd.nReac, sacadoStorageDimension);
    w += kmcd.nReac*len;
    auto kfor = value_type_1d_view_type(w, kmcd.nReac, sacadoStorageDimension);
    w += kmcd.nReac*len;
    auto krev = value_type_1d_view_type(w, kmcd.nReac, sacadoStorageDimension);
    w += kmcd.nReac*len;
    auto ropFor = value_type_1d_view_type(w, kmcd.nReac, sacadoStorageDimension);
    w += kmcd.nReac*len;
    auto ropRev = value_type_1d_view_type(w, kmcd.nReac, sacadoStorageDimension);
    w += kmcd.nReac*len;
    auto Crnd = value_type_1d_view_type(w, kmcd.nReac, sacadoStorageDimension);
    w += kmcd.nReac*len;


    const ordinal_type work_kfor_rev_size =
    Impl::KForwardReverse<value_type,device_type>::getWorkSpaceSize(kmcd);
    auto work_kfor_rev = value_type_1d_view_type(w, work_kfor_rev_size, sacadoStorageDimension);
    w += work_kfor_rev_size*len;

    auto iter = Kokkos::View<ordinal_type*,
                             Kokkos::LayoutRight,
                             typename real_type_1d_view_type::memory_space>(
      (ordinal_type*)w, kmcd.nReac * 2);
    w += kmcd.nReac * 2;

    using range_type = Kokkos::pair<ordinal_type, ordinal_type>;

    const value_type_1d_view_type omega_t = Kokkos::subview(omega,range_type(0, 1));
    const value_type_1d_view_type omega_s = Kokkos::subview(omega,range_type(1, kmcd.nSpec + 1));

    team_invoke_detail(member,
                       t,
                       p,
                       density,
                       Ys,
                       omega_t,
                       omega_s,
                       /// workspace
                       gk,
                       hks,
                       cpks,
                       concX,
                       concM,
                       kfor,
                       krev,
                       ropFor,
                       ropRev,
                       Crnd,
                       iter,
                       work_kfor_rev,
                       kmcd);
  }

  template<typename MemberType,typename WorkViewType >
  KOKKOS_FORCEINLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const real_type& t,
    const real_type& p,
    const real_type& density,
    const real_type_1d_view_type& Ys, /// (kmcd.nSpec)
    /// output
    const real_type_1d_view_type& omega, /// (kmcd.nSpec + 1)
    /// workspace
    const WorkViewType& work,
    /// kmcd.nSpec(5) : Xc, concX,gk,hks,cpks,
    /// kmcd.nReac(5) : concM,kfor,krev,rop,Crnd
    /// const input from kinetic model
    const kinetic_model_type& kmcd)
  {
    // const real_type zero(0);

    ///
    /// workspace needed gk, hks, kfor, krev
    ///
    auto w = (real_type*)work.data();

    // auto Xc = RealType1DViewType(w, kmcd.nSpec);
    // w += kmcd.nSpec;
    auto gk = real_type_1d_view_type(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto hks = real_type_1d_view_type(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto cpks = real_type_1d_view_type(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto concX = real_type_1d_view_type(w, kmcd.nSpec);
    w += kmcd.nSpec;

    auto concM = real_type_1d_view_type(w, kmcd.nReac);
    w += kmcd.nReac;
    auto kfor = real_type_1d_view_type(w, kmcd.nReac);
    w += kmcd.nReac;
    auto krev = real_type_1d_view_type(w, kmcd.nReac);
    w += kmcd.nReac;
    auto ropFor = real_type_1d_view_type(w, kmcd.nReac);
    w += kmcd.nReac;
    auto ropRev = real_type_1d_view_type(w, kmcd.nReac);
    w += kmcd.nReac;
    auto Crnd = real_type_1d_view_type(w, kmcd.nReac);
    w += kmcd.nReac;

    auto iter = Kokkos::View<ordinal_type*,
                             Kokkos::LayoutRight,
                             typename real_type_1d_view_type::memory_space>(
      (ordinal_type*)w, kmcd.nReac * 2);
    w += kmcd.nReac * 2;

    const ordinal_type work_kfor_rev_size =
    Impl::KForwardReverse<real_type,device_type>::getWorkSpaceSize(kmcd);
    auto work_kfor_rev = real_type_1d_view_type(w, work_kfor_rev_size);
    w += work_kfor_rev_size;

    auto rhs = (real_type*)omega.data();
    auto omega_t = real_type_1d_view_type(rhs, 1);
    rhs += 1;
    auto omega_s = real_type_1d_view_type(rhs, kmcd.nSpec);
    rhs += kmcd.nSpec;

    team_invoke_detail(member,
                       t,
                       p,
                       density,
                       Ys,
                       omega_t,
                       omega_s,
                       /// workspace
                       gk,
                       hks,
                       cpks,
                       concX,
                       concM,
                       kfor,
                       krev,
                       ropFor,
                       ropRev,
                       Crnd,
                       iter,
                       work_kfor_rev,
                       kmcd);
  }
};


} // namespace Impl
} // namespace TChem

#endif
