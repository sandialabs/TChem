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
#ifndef __TCHEM_IMPL_RATEOFPROGRESS_SURFACE_IND_HPP__
#define __TCHEM_IMPL_RATEOFPROGRESS_SURFACE_IND_HPP__
// Compute rate of progess Forward and reverse
// Use this function when compute rate of progess from another application
// example csp
#include "TChem_Impl_Gk.hpp"
#include "TChem_Impl_KForwardReverseSurface.hpp"
#include "TChem_Impl_MolarConcentrations.hpp"
#include "TChem_Impl_RateOfProgressSurface.hpp"
#include "TChem_Util.hpp"

namespace TChem {
namespace Impl {

struct RateOfProgressSurfaceInd
{

  // template<typename KineticModelConstDataType,
  //         typename KineticModelConstSurfDataType>
  // KOKKOS_INLINE_FUNCTION
  // static
  // ordinal_type getWorkSpaceSize(const KineticModelConstDataType &kmcd,
  // const KineticModelConstSurfDataType &kmcdSurf) {
  // return (4*kmcd.nSpec + 4*kmcdSurf.nSpec+ 4*kmcdSurf.nReac );
  // }
  template<typename MemberType,
           typename RealType1DViewType,
           typename OrdinalType1DViewType,
           typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    /// input
    const real_type& t,
    const real_type& p,
    const RealType1DViewType& Yk, /// (kmcd.nSpec)
    const RealType1DViewType& zSurf,
    /// output
    const RealType1DViewType& ropFor,
    const RealType1DViewType& ropRev,
    /// workspace
    // gas species
    const RealType1DViewType& gk,
    const RealType1DViewType& hks,
    const RealType1DViewType& cpks,
    // surface species
    const RealType1DViewType& Surf_gk,
    const RealType1DViewType& Surf_hks,
    const RealType1DViewType& Surf_cpks,

    const RealType1DViewType& concX,
    const RealType1DViewType& concXSurf,
    // const RealType1DViewType &concM,
    const RealType1DViewType& kfor,
    const RealType1DViewType& krev,

    // const RealType1DViewType &Crnd,
    const OrdinalType1DViewType& iter,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd,
    /// const input from surface kinetic model
    const KineticSurfModelConstDataType& kmcdSurf)
  {
    ///const real_type zero(0); ///not used
    const real_type one(1);
    const real_type ten(10);

    // is Xc  mass fraction or concetration ?

    /* 1. compute molar concentrations from mass fraction (moles/cm3) */
    MolarConcentrations::team_invoke(member,
                                     t,
                                     p,
                                     Yk, // need to be mass fraction
                                     concX,
                                     kmcd);

    /// 2. initialize and transform molar concentrations (kmol/m3) to
    /// (moles/cm3)
    member.team_barrier();
    {
      const real_type one_e_minus_three(1e-3);
      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, kmcd.nSpec),
        [&](const ordinal_type& i) { concX(i) *= one_e_minus_three; });
    }

    /* 2. surface molar concentrations (moles/cm2) */
    /* FR: note there is one type of site, and site occupancy \sigma_k =1.0 */
    {
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, kmcdSurf.nSpec),
                           [&](const ordinal_type& i) {
                             concXSurf(i) =
                               zSurf(i) * kmcdSurf.sitedensity / one;
                           });
    }

    /*3. compute (-ln(T)+dS/R-dH/RT) for each species */
    /* We are computing (dS-dH)/R for each species
    gk is Gibbs free energy */

    GkSurfGas ::team_invoke(member,
                            t, /// input
                            gk,
                            hks,  /// output
                            cpks, /// workspace
                            kmcd);
    // member.team_barrier();

    // surfaces
    GkSurfGas ::team_invoke(member,
                            t, /// input
                            Surf_gk,
                            Surf_hks,  /// output
                            Surf_cpks, /// workspace
                            kmcdSurf);

    member.team_barrier();

    /* compute forward and reverse rate constants */
    KForwardReverseSurface ::team_invoke(member,
                                         t,
                                         p,
                                         gk,
                                         Surf_gk, /// input
                                         kfor,
                                         krev, /// output
                                         iter,
                                         kmcd,    // gas info
                                         kmcdSurf // surface info
    );
    member.team_barrier();

    // /// compute rate-of-progress
    RateOfProgressSurface::team_invoke(member,
                                       kfor,
                                       krev,
                                       concX,
                                       concXSurf, /// input
                                       ropFor,
                                       ropRev, /// output
                                       iter,   /// workspace for iterators
                                       kmcdSurf);

    member.team_barrier();

    /* transform from mole/(cm2.s) to kmol/(m2.s) */
    /* check this unit convection*/

    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, kmcdSurf.nReac),
                         [&](const ordinal_type& i) {
                           ropFor(i) *= ten;
                           ropRev(i) *= ten;
                         });
  }

  template<typename MemberType,
           typename WorkViewType,
           typename RealType1DViewType,
           typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType>

  KOKKOS_FORCEINLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const real_type& t,
    const real_type& p,
    const RealType1DViewType& Yk,    /// (kmcd.nSpec)
    const RealType1DViewType& zSurf, //(kmcdSurf.nSpec)

    /// output
    const RealType1DViewType& ropFor, /// (kmcd.nSpec)
    const RealType1DViewType& ropRev,
    /// workspace
    const WorkViewType& work,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf)
  {
    ///const real_type zero(0); /// not used

    ///
    /// workspace needed gk, hks, kfor, krev
    ///
    auto w = (real_type*)work.data();

    // gas species thermal properties // uses a different model for that gas
    // phase
    auto gk = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto hks = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto cpks = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto concX = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;

    auto kfor = RealType1DViewType(w, kmcdSurf.nReac);
    w += kmcdSurf.nReac;
    auto krev = RealType1DViewType(w, kmcdSurf.nReac);
    w += kmcdSurf.nReac;

    // surface species thermal properties
    auto Surf_gk = RealType1DViewType(w, kmcdSurf.nSpec);
    w += kmcdSurf.nSpec;
    auto Surf_hks = RealType1DViewType(w, kmcdSurf.nSpec);
    w += kmcdSurf.nSpec;
    auto Surf_cpks = RealType1DViewType(w, kmcdSurf.nSpec);
    w += kmcdSurf.nSpec;
    auto concXSurf = RealType1DViewType(w, kmcdSurf.nSpec);
    w += kmcdSurf.nSpec;

    auto iter = Kokkos::View<ordinal_type*,
                             Kokkos::LayoutRight,
                             typename WorkViewType::memory_space>(
      (ordinal_type*)w, kmcdSurf.nReac * 2);
    w += kmcdSurf.nReac * 2;

    team_invoke_detail(member,
                       t,
                       p,
                       Yk,
                       zSurf,
                       ropFor,
                       ropRev,
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
                       iter,
                       kmcd,
                       kmcdSurf);
  }
};

} // namespace Impl
} // namespace TChem

#endif
