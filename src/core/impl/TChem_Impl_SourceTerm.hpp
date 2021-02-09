/* =====================================================================================
TChem version 2.1.0
Copyright (2020) NTESS
https://github.com/sandialabs/TChem

Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains 
certain rights in this software.

This file is part of TChem. TChem is open-source software: you can redistribute it
and/or modify it under the terms of BSD 2-Clause License
(https://opensource.org/licenses/BSD-2-Clause). A copy of the license is also
provided under the main directory

Questions? Contact Cosmin Safta at <csafta@sandia.gov>, or
           Kyungjoo Kim at <kyukim@sandia.gov>, or
           Oscar Diaz-Ibarra at <odiazib@sandia.gov>

Sandia National Laboratories, Livermore, CA, USA
===================================================================================== */


#ifndef __TCHEM_IMPL_SOURCE_TERM_HPP__
#define __TCHEM_IMPL_SOURCE_TERM_HPP__

#include "TChem_Impl_CpMixMs.hpp"
#include "TChem_Impl_EnthalpySpecMs.hpp"
#include "TChem_Impl_MolarConcentrations.hpp"
#include "TChem_Impl_ReactionRates.hpp"
#include "TChem_Impl_RhoMixMs.hpp"
#include "TChem_Util.hpp"

namespace TChem {
namespace Impl {

struct SourceTerm
{
  template<typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd)
  {
    const ordinal_type workspace_size = (5 * kmcd.nSpec + 8 * kmcd.nReac);
    return workspace_size;
  }

  ///
  ///  \param t  : temperature [K]
  ///  \param Xc : array of \f$N_{spec}\f$ doubles \f$((XC_1,XC_2,...,XC_N)\f$:
  ///              molar concentrations XC \f$[kmol/m^3]\f$
  ///  \return omega : array of \f$N_{spec}\f$ molar reaction rates
  ///  \f$\dot{\omega}_i\f$ \f$\left[kmol/(m^3\cdot s)\right]\f$
  ///
  template<typename MemberType,
           typename RealType1DViewType,
           typename OrdinalType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    /// input
    const real_type& t,
    const real_type& p,
    const RealType1DViewType& Ys, /// (kmcd.nSpec)
    /// output
    const RealType1DViewType& omega_t, /// (1)
    const RealType1DViewType& omega,   /// (kmcd.nSpec)
    /// workspace
    const RealType1DViewType& Xc,
    const RealType1DViewType& gk,
    const RealType1DViewType& hks,
    const RealType1DViewType& cpks,
    const RealType1DViewType& concX,
    const RealType1DViewType& concM,
    const RealType1DViewType& kfor,
    const RealType1DViewType& krev,
    const RealType1DViewType& ropFor,
    const RealType1DViewType& ropRev,
    const RealType1DViewType& Crnd,
    const OrdinalType1DViewType& iter,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    const real_type zero(0), one(1);

    /// 0. convert Ys to Xc
    // MolarConcentrations
    //   ::team_invoke(member,
    //                 t, p, Ys,
    //                 Xc,
    //                 kmcd);

    /// 1. compute molar reaction rates
    ReactionRates ::team_invoke_detail(member,
                                       t,
                                       p,
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
                                       kmcd);

    /// 2. transform molar reaction rates to mass reaction rates
    Kokkos::parallel_for(
      Kokkos::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& i) { omega(i) *= kmcd.sMass(i); });

    /// 3. compute density, cpmix
    const real_type rhomix = RhoMixMs::team_invoke(member, t, p, Ys, kmcd);
    const real_type cpmix = CpMixMs::team_invoke(member, t, Ys, cpks, kmcd);

    /// 4. compute species enthalies
    EnthalpySpecMs ::team_invoke(member, t, hks, cpks, kmcd);

    /// 5. transform reaction rates to source term (Wi/rho)
    const real_type orho = one / rhomix;
    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, kmcd.nSpec),
                         [&](const ordinal_type& i) { omega(i) *= orho; });

    /// 6. compute source term for temperature
    Kokkos::single(Kokkos::PerTeam(member), [&]() { omega_t(0) = zero; });
    member.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, kmcd.nSpec),
                         [&](const ordinal_type& i) {
                           const real_type val = -omega(i) * hks(i);
                           Kokkos::atomic_fetch_add(&omega_t(0), val);
                         });
    member.team_barrier();
    Kokkos::single(Kokkos::PerTeam(member), [&]() { omega_t(0) /= cpmix; });
  }

  template<typename MemberType,
           typename WorkViewType,
           typename RealType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_FORCEINLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const real_type& t,
    const real_type& p,
    const RealType1DViewType& Ys, /// (kmcd.nSpec)
    /// output
    const RealType1DViewType& omega, /// (kmcd.nSpec + 1)
    /// workspace
    const WorkViewType& work,
    /// kmcd.nSpec(5) : Xc, concX,gk,hks,cpks,
    /// kmcd.nReac(5) : concM,kfor,krev,rop,Crnd
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    // const real_type zero(0);

    ///
    /// workspace needed gk, hks, kfor, krev
    ///
    auto w = (real_type*)work.data();

    auto Xc = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto gk = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto hks = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto cpks = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto concX = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;

    auto concM = RealType1DViewType(w, kmcd.nReac);
    w += kmcd.nReac;
    auto kfor = RealType1DViewType(w, kmcd.nReac);
    w += kmcd.nReac;
    auto krev = RealType1DViewType(w, kmcd.nReac);
    w += kmcd.nReac;
    auto ropFor = RealType1DViewType(w, kmcd.nReac);
    w += kmcd.nReac;
    auto ropRev = RealType1DViewType(w, kmcd.nReac);
    w += kmcd.nReac;
    auto Crnd = RealType1DViewType(w, kmcd.nReac);
    w += kmcd.nReac;

    auto iter = Kokkos::View<ordinal_type*,
                             Kokkos::LayoutRight,
                             typename WorkViewType::memory_space>(
      (ordinal_type*)w, kmcd.nReac * 2);
    w += kmcd.nReac * 2;

    auto omega_t = RealType1DViewType(omega.data(), 1);
    auto omega_s = RealType1DViewType(omega.data() + 1, kmcd.nSpec);

    team_invoke_detail(member,
                       t,
                       p,
                       Ys,
                       omega_t,
                       omega_s,
                       /// workspace
                       Xc,
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
                       kmcd);
  }
};

} // namespace Impl
} // namespace TChem

#endif
