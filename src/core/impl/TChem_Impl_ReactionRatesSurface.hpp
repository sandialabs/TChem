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
#ifndef __TCHEM_IMPL_ReactionRatesSurface_HPP__
#define __TCHEM_IMPL_ReactionRatesSurface_HPP__

#include "TChem_Impl_Gk.hpp"
#include "TChem_Impl_KForwardReverseSurface.hpp"
#include "TChem_Impl_MolarConcentrations.hpp"
#include "TChem_Impl_RateOfProgressSurface.hpp"
#include "TChem_Util.hpp"

namespace TChem {
namespace Impl {

template<typename ValueType, typename DeviceType>
struct ReactionRatesSurface
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
  using kinetic_surf_model_type = KineticSurfModelConstData<device_type>;

  KOKKOS_INLINE_FUNCTION static ordinal_type getWorkSpaceSize(
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf)
  {
    return (4 * kmcd.nSpec + 0 * kmcd.nReac + 4 * kmcdSurf.nSpec +
            6 * kmcdSurf.nReac);
  }

  /**
  \param scal : array of \f$N_{spec}+1\f$ doubles \f$((T,XC_1,XC_2,...,XC_N)\f$:
            temperature T [K], molar concentrations XC \f$[kmol/m^3]\f$
  \param Nvars : no. of variables \f$N_{vars}=N_{spec}+1\f$
  \return omega : array of \f$N_{spec}\f$ molar reaction rates
  \f$\dot{\omega}_i\f$ \f$\left[kmol/(m^3\cdot s)\right]\f$
  */

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    /// input
    const value_type& t,
    const value_type& p,
    const value_type& density,
    const value_type_1d_view_type& Yk, /// (kmcd.nSpec)
    const value_type_1d_view_type& zSurf,
    /// output
    const value_type_1d_view_type& omega,     /// (kmcd.nSpec)
    const value_type_1d_view_type& omegaSurf, /// (kmcd.nSpec)
    /// workspace
    // gas species
    const value_type_1d_view_type& gk,
    const value_type_1d_view_type& hks,
    const value_type_1d_view_type& cpks,
    // surface species
    const value_type_1d_view_type& Surf_gk,
    const value_type_1d_view_type& Surf_hks,
    const value_type_1d_view_type& Surf_cpks,

    const value_type_1d_view_type& concX,
    const value_type_1d_view_type& concXSurf,
    // const RealType1DViewType &concM,
    const value_type_1d_view_type& kfor,
    const value_type_1d_view_type& krev,
    const value_type_1d_view_type& ropFor,
    const value_type_1d_view_type& ropRev,
    // const RealType1DViewType &Crnd,
    const value_type_1d_view_type& CoverageFactor,
    /// const input from kinetic model
    const kinetic_model_type& kmcd,
    /// const input from surface kinetic model
    const kinetic_surf_model_type& kmcdSurf)
  {
    const real_type zero(0);
    const real_type one(1);
    const real_type ten(10);

    using MolarConcentrations = MolarConcentrations<value_type,device_type>;
    using GkSurfGas = GkFcnSurfGas<value_type,device_type>;
    using KForwardReverseSurface = KForwardReverseSurface<value_type,device_type>;
    using RateOfProgressSurface = RateOfProgressSurface<value_type,device_type>;

    // is Xc  mass fraction or concetration ?

    /* 1. compute molar concentrations from mass fraction (moles/cm3) */
    MolarConcentrations::team_invoke(member,
                                     t,
                                     p,
                                     density,
                                     Yk, // need to be mass fraction
                                     concX,
                                     kmcd);

    /// 2. initialize and transform molar concentrations (kmol/m3) to
    /// (moles/cm3)
    member.team_barrier();
    {
      const real_type one_e_minus_three(1e-3);
      Kokkos::parallel_for(Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
                           [&](const ordinal_type& i) {
                             omega(i) = zero;
                             concX(i) *= one_e_minus_three;
                           });
    }

    /* 2. surface molar concentrations (moles/cm2) */
    /* FR: note there is one type of site, and site occupancy \sigma_k =1.0 */
    {
      Kokkos::parallel_for(Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcdSurf.nSpec),
                           [&](const ordinal_type& i) {
                             concXSurf(i) =
                               zSurf(i) * kmcdSurf.sitedensity / one;
                             omegaSurf(i) = zero;
                           });
    }

    /*3. compute (-ln(T)+dS/R-dH/RT) for each species */
    /* We are computing (dS-dH)/R for each species
    gk is Gibbs free energy */

    GkSurfGas::team_invoke(member,
                            t, /// input
                            gk,
                            hks,  /// output
                            cpks, /// workspace
                            kmcd);
    // member.team_barrier();

    // surfaces
    GkSurfGas::team_invoke(member,
                            t, /// input
                            Surf_gk,
                            Surf_hks,  /// output
                            Surf_cpks, /// workspace
                            kmcdSurf);

    member.team_barrier();

    /* compute forward and reverse rate constants */
    KForwardReverseSurface::team_invoke(member,
                                         t,
                                         p,
                                         gk,
                                         Surf_gk, /// input
                                         kfor,
                                         krev, /// output
                                         kmcd,    // gas info
                                         kmcdSurf // surface info
    );
    member.team_barrier();

    // /// compute rate-of-progress
    RateOfProgressSurface::team_invoke_detail(member,
                                       t,
                                       kfor,
                                       krev,
                                       concX,
                                       concXSurf, /// input
                                       ropFor,
                                       ropRev, /// output
                                       CoverageFactor,   /// workspace for iterators
                                       kmcdSurf);

    member.team_barrier();

    /* assemble reaction rates */
    Kokkos::parallel_for(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcdSurf.nReac),
      [&](const ordinal_type& i) {
        for (ordinal_type j = 0; j < kmcdSurf.reacNreac(i); ++j) {
          const value_type val =
            kmcdSurf.reacNuki(i, j) * (ropFor(i) - ropRev(i));
          const ordinal_type kspec = kmcdSurf.reacSidx(i, j); // species index
          if (kmcdSurf.reacSsrf(i, j) == 1) {                 // surface
            // omegaSurf(kspec) += kmcdSurf.reacNuki(i,j) *
            // (ropFor(i)-ropRev(i));
            Kokkos::atomic_add(&omegaSurf(kspec), val);
          } else { // gas
            Kokkos::atomic_add(&omega(kspec), val);
            // omega(kspec) += kmcdSurf.reacNuki(i,j) * (ropFor(i)-ropRev(i));
          }
        }

        for (ordinal_type j = 0; j < kmcdSurf.reacNprod(i); j++) {

          const ordinal_type joff = kmcdSurf.maxSpecInReac / 2;
          const ordinal_type kspec =
            kmcdSurf.reacSidx(i, j + joff); // species index
          const value_type val =
            kmcdSurf.reacNuki(i, j + joff) * (ropFor(i) - ropRev(i));
          if (kmcdSurf.reacSsrf(i, j + joff) == 1) { // surface
            Kokkos::atomic_add(&omegaSurf(kspec), val);
            // omegaSurf(kspec) += kmcdSurf.reacNuki(i,j+joff) *
            // (ropFor(i)-ropRev(i));
          } else { // gas
            Kokkos::atomic_add(&omega(kspec), val);
            // omega(kspec) += kmcdSurf.reacNuki(i,j+joff) *
            // (ropFor(i)-ropRev(i));
          }
        }
      }); /* done loop over reactions */

    /* transform from mole/(cm2.s) to kmol/(m2.s) */

    Kokkos::parallel_for(Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcdSurf.nSpec),
                         [&](const ordinal_type& i) { omegaSurf(i) *= ten; });

    Kokkos::parallel_for(Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
                         [&](const ordinal_type& i) { omega(i) *= ten; });
  }

  template<typename MemberType,
           typename WorkViewType>

  KOKKOS_FORCEINLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const real_type& t,
    const real_type& p,
    const real_type& density,
    const real_type_1d_view_type& Yk,    /// (kmcd.nSpec)
    const real_type_1d_view_type& zSurf, //(kmcdSurf.nSpec)

    /// output
    const real_type_1d_view_type& omega, /// (kmcd.nSpec)
    const real_type_1d_view_type& omegaSurf,
    /// workspace
    const WorkViewType& work,
    /// kmcd.nSpec(4) : concX,gk,hks,cpks,
    /// kmcd.nReac(5) : concM,kfor,krev,rop,Crnd
    /// const input from kinetic model
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf)
  {
    ///const real_type zero(0); // not used

    ///
    /// workspace needed gk, hks, kfor, krev
    ///
    auto w = (real_type*)work.data();

    // gas species thermal properties // uses a different model for that gas
    // phase
    auto gk = real_type_1d_view_type(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto hks = real_type_1d_view_type(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto cpks = real_type_1d_view_type(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto concX = real_type_1d_view_type(w, kmcd.nSpec);
    w += kmcd.nSpec;

    // auto concM  = RealType1DViewType(w, kmcd.nReac); w+=kmcd.nReac;
    auto kfor = real_type_1d_view_type(w, kmcdSurf.nReac);
    w += kmcdSurf.nReac;
    auto krev = real_type_1d_view_type(w, kmcdSurf.nReac);
    w += kmcdSurf.nReac;
    auto ropFor = real_type_1d_view_type(w, kmcdSurf.nReac);
    w += kmcdSurf.nReac;
    auto ropRev = real_type_1d_view_type(w, kmcdSurf.nReac);
    w += kmcdSurf.nReac;
    // auto Crnd   = RealType1DViewType(w, kmcd.nReac); w+=kmcd.nReac;

    // surface species thermal properties
    auto Surf_gk = real_type_1d_view_type(w, kmcdSurf.nSpec);
    w += kmcdSurf.nSpec;
    auto Surf_hks = real_type_1d_view_type(w, kmcdSurf.nSpec);
    w += kmcdSurf.nSpec;
    auto Surf_cpks = real_type_1d_view_type(w, kmcdSurf.nSpec);
    w += kmcdSurf.nSpec;
    auto concXSurf = real_type_1d_view_type(w, kmcdSurf.nSpec);
    w += kmcdSurf.nSpec;

    auto CoverageFactor = real_type_1d_view_type(w, kmcdSurf.nReac);
    w += kmcdSurf.nReac;

    team_invoke_detail(member,
                       t,
                       p,
                       density,
                       Yk,
                       zSurf,
                       omega,
                       omegaSurf,
                       gk,
                       hks,
                       cpks,
                       Surf_gk,
                       Surf_hks,
                       Surf_cpks,
                       concX,
                       concXSurf,
                       kfor,
                       krev,
                       ropFor,
                       ropRev,
                       CoverageFactor,
                       kmcd,
                       kmcdSurf);
  }
};

} // namespace Impl
} // namespace TChem

#endif
