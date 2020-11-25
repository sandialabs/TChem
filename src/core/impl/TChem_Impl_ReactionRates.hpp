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
#ifndef __TCHEM_IMPL_REACTIONRATES_HPP__
#define __TCHEM_IMPL_REACTIONRATES_HPP__

#include "TChem_Impl_Crnd.hpp"
#include "TChem_Impl_Gk.hpp"
#include "TChem_Impl_KForwardReverse.hpp"
#include "TChem_Impl_MolarConcentrations.hpp"
#include "TChem_Impl_RateOfProgress.hpp"
#include "TChem_Impl_ThirdBodyConcentrations.hpp"
#include "TChem_Util.hpp"
// #define TCHEM_ENABLE_SERIAL_TEST_OUTPUT
namespace TChem {
namespace Impl {

struct ReactionRates
{
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
    const RealType1DViewType& Ys, /// (kmcd.nSpec) mass fraction
    /// output
    const RealType1DViewType& omega, /// (kmcd.nSpec)
    /// workspace
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
    const real_type zero(0);

    /// 0. compute (-ln(T)+dS/R-dH/RT) for each species
    Gk ::team_invoke(member,
                     t, /// input
                     gk,
                     hks,  /// output
                     cpks, /// workspace
                     kmcd);
    member.team_barrier();

    /// 1. compute forward and reverse rate constants
    KForwardReverse ::team_invoke(member,
                                  t,
                                  p,
                                  gk, /// input
                                  kfor,
                                  krev, /// output
                                  iter,
                                  kmcd);

    ///
    /// workspace needed concX = w0, concM = w1, kfor, krev
    ///
    /* 1. compute molar concentrations from mass fraction (moles/cm3) */
    MolarConcentrations::team_invoke(member,
                                     t,
                                     p,
                                     Ys, // need to be mass fraction
                                     concX,
                                     kmcd);
    member.team_barrier();
    // / 2. initialize and transform molar concentrations (kmol/m3) to
    // (moles/cm3)
    {
      const real_type one_e_minus_three(1e-3);
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, kmcd.nSpec),
                           [&](const ordinal_type& i) {
                             omega(i) = zero;
                             concX(i) = concX(i) * one_e_minus_three;
                             // concX(i) = Xc(i)*one_e_minus_three;
                           });
    }
    member.team_barrier();
    /// 3. get 3rd-body concentrations
    ThirdBodyConcentrations ::team_invoke(member,
                                          concX,
                                          concM, /// output
                                          kmcd);
    member.team_barrier();

    /// 4. compute rate-of-progress
    RateOfProgress::team_invoke(member,
                                kfor,
                                krev,
                                concX, /// input
                                ropFor,
                                ropRev, /// output
                                iter,   /// workspace for iterators
                                kmcd);

    /// 5. compute pressure dependent factors
    Crnd::team_invoke(member,
                      t,
                      kfor,
                      concX,
                      concM, /// input
                      Crnd,  /// output
                      iter,
                      kmcd);
    member.team_barrier();

    /// 6. update rop with Crnd and assemble reaction rates
    auto rop = ropFor;
    Kokkos::parallel_for(
      Kokkos::TeamVectorRange(member, kmcd.nReac), [&](const ordinal_type& i) {
        rop(i) -= ropRev(i);
        rop(i) *= Crnd(i);
        const real_type rop_at_i = rop(i);
        for (ordinal_type j = 0; j < kmcd.reacNreac(i); ++j) {
          const ordinal_type kspec = kmcd.reacSidx(i, j);
          // omega(kspec) += kmcd.reacNuki(i,j)*rop_at_i;
          const real_type val = kmcd.reacNuki(i, j) * rop_at_i;
          Kokkos::atomic_fetch_add(&omega(kspec), val);
        }
        const ordinal_type joff = kmcd.reacSidx.extent(1) / 2;
        for (ordinal_type j = 0; j < kmcd.reacNprod(i); ++j) {
          const ordinal_type kspec = kmcd.reacSidx(i, j + joff);
          // omega(kspec) += kmcd.reacNuki(i,j+joff)*rop_at_i;
          const real_type val = kmcd.reacNuki(i, j + joff) * rop_at_i;
          Kokkos::atomic_fetch_add(&omega(kspec), val);
        }
      });

    /// 8. check for reactions with real stoichiometric coefficients
    if (kmcd.nRealNuReac > 0) {
      member.team_barrier();
      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, kmcd.nRealNuReac),
        [&](const ordinal_type& ir) {
          const ordinal_type i = kmcd.reacRnu(ir);
          const real_type rop_at_i = rop(i);
          for (ordinal_type j = 0; j < kmcd.reacNreac(i); ++j) {
            const ordinal_type kspec = kmcd.reacSidx(i, j);
            // omega(kspec) += kmcd.reacRealNuki(ir,j)*rop_at_i;
            const real_type val = kmcd.reacRealNuki(ir, j) * rop_at_i;
            Kokkos::atomic_fetch_add(&omega(kspec), val);
          }
          const ordinal_type joff = kmcd.reacSidx.extent(1) / 2;
          const ordinal_type joff2 = kmcd.reacNreac(i);
          for (ordinal_type j = 0; j < kmcd.reacNprod(i); ++j) {
            const ordinal_type kspec = kmcd.reacSidx(i, j + joff);
            // omega(kspec) += kmcd.reacRealNuki(i,j)*rop_at_i;
            const real_type val = kmcd.reacRealNuki(ir, j+joff2) * rop_at_i;
            Kokkos::atomic_fetch_add(&omega(kspec), val);
          }
        });
    }
    member.team_barrier();

    /// 9. transform from mole/(cm3.s) to kmol/(m3.s)
    {
      const real_type one_e_3(1e3);
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, kmcd.nSpec),
                           [&](const ordinal_type& i) { omega(i) *= one_e_3; });
    }
#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("ReactionRates.team_invoke.test.out", "a+");
      fprintf(fs, ":: ReactionRates::team_invoke\n");
      fprintf(fs, ":::: input\n");
      fprintf(fs,
              "     nSpec %3d, nReac %3d, t %e, p %e\n",
              kmcd.nSpec,
              kmcd.nReac,
              t,
              p);
      fprintf(fs, ":: Crnd\n");
      for (int i = 0; i < int(Crnd.extent(0)); ++i)
        fprintf(fs, "     i %3d, Crndr %e\n", i, Crnd(i));
      fprintf(fs, "\n");
      fprintf(fs, ":: Crnd*kfor\n");
      for (int i = 0; i < int(Crnd.extent(0)); ++i)
        fprintf(fs,
                "     i %3d, Crnd*kfor %e\n",
                i,
                real_type(1e-3) * Crnd(i) * kfor(i));
      fprintf(fs, ":: Crnd*krev\n");
      for (int i = 0; i < int(Crnd.extent(0)); ++i)
        fprintf(fs,
                "     i %3d, Crnd*krev %e\n",
                i,
                real_type(1e-3) * Crnd(i) * krev(i));
      fprintf(fs, "\n");
      fprintf(fs, ":::: output\n");
      for (int i = 0; i < int(omega.extent(0)); ++i)
        fprintf(fs, "     i %3d, omega %e\n", i, omega(i));
      fprintf(fs, "\n");
    }
#endif

    // if (member.league_rank() == 0) {
    //   FILE *fs = fopen("ReactionRates.team_invoke.test.out", "a+");
    //   for (int i=0;i<int(Crnd.extent(0));++i)
    //     fprintf(fs, " %d %e\n", i , real_type(1e-3)*Crnd(i)*kfor(i));
    // }
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
    const RealType1DViewType& omega, /// (kmcd.nSpec)
    /// workspace
    const WorkViewType& work,
    /// kmcd.nSpec(4) : concX,gk,hks,cpks,
    /// kmcd.nReac(5) : concM,kfor,krev,rop,Crnd
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    ///const real_type zero(0);  /// not used

    ///
    /// workspace needed gk, hks, kfor, krev
    ///
    auto w = (real_type*)work.data();

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

    team_invoke_detail(member,
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
  }
};

} // namespace Impl
} // namespace TChem

#endif
