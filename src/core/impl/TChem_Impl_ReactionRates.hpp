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

template<typename ValueType, typename DeviceType>
struct ReactionRates
{
  ///
  ///  \param t  : temperature [K]
  ///  \param Xc : array of \f$N_{spec}\f$ doubles \f$((XC_1,XC_2,...,XC_N)\f$:
  ///              molar concentrations XC \f$[kmol/m^3]\f$
  ///  \return omega : array of \f$N_{spec}\f$ molar reaction rates
  ///  \f$\dot{\omega}_i\f$ \f$\left[kmol/(m^3\cdot s)\right]\f$
  ///
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

    return (4 * kmcd.nSpec + 8 * kmcd.nReac + work_kfor_rev_size);
  }

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    /// input
    const value_type& t,
    const value_type& p,
    const value_type& density,
    const value_type_1d_view_type& Ys, /// (kmcd.nSpec) mass fraction
    /// output
    const value_type_1d_view_type& omega, /// (kmcd.nSpec)
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
    using Gk = GkFcn<value_type,device_type>;
    using KForwardReverse = KForwardReverse<value_type,device_type>;
    using MolarConcentrations = MolarConcentrations<value_type,device_type>;
    using ThirdBodyConcentrations = ThirdBodyConcentrations<value_type,device_type>;
    using RateOfProgress = RateOfProgress<value_type,device_type>;
    using Crnd_type = Impl::Crnd<value_type,device_type>;

    const real_type zero(0);

    /// 0. compute (-ln(T)+dS/R-dH/RT) for each species
    Gk::team_invoke(member,
                     t, /// input
                     gk,
                     hks,  /// output
                     cpks, /// workspace
                     kmcd);
    member.team_barrier();

    /// 1. compute forward and reverse rate constants
    KForwardReverse::team_invoke(member,
                                  t,
                                  p,
                                  gk, /// input
                                  kfor,
                                  krev, /// output
                                  iter,
                                  work_kfor_rev,
                                  kmcd);

    ///
    /// workspace needed concX = w0, concM = w1, kfor, krev
    ///
    /* 1. compute molar concentrations from mass fraction (moles/cm3) */
    MolarConcentrations::team_invoke(member,
                                     t,
                                     p,
                                     density,
                                     Ys, // need to be mass fraction
                                     concX,
                                     kmcd);
    //

    member.team_barrier();
    // / 2. initialize and transform molar concentrations (kmol/m3) to
    // (moles/cm3)
    {
      const real_type one_e_minus_three(1e-3);
      Kokkos::parallel_for(Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
                           [&](const ordinal_type& i) {
                             omega(i) = zero;
                             concX(i) = concX(i) * one_e_minus_three;
                             // concX(i) = Xc(i)*one_e_minus_three;
                           });
    }
    member.team_barrier();
    // /// 3. get 3rd-body concentrations
    ThirdBodyConcentrations::team_invoke(member,
                                          concX,
                                          concM, /// output
                                          kmcd);
    member.team_barrier();
    //
    // /// 4. compute rate-of-progress
    RateOfProgress::team_invoke(member,
                                kfor,
                                krev,
                                concX, /// input
                                ropFor,
                                ropRev, /// output
                                iter,   /// workspace for iterators
                                kmcd);
    //
    // /// 5. compute pressure dependent factors
    Crnd_type::team_invoke(member,
                      t,
                      kfor,
                      concX,
                      concM, /// input
                      Crnd,  /// output
                      iter,
                      kmcd);
    member.team_barrier();

    /// 6. update rop with Crnd and assemble reaction rates
#if defined(SACADO_VIEW_CUDA_HIERARCHICAL)
    Kokkos::abort("This is not ready yet");
#else
    Kokkos::parallel_for
      (Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nReac),
       [&](const ordinal_type& i) {
        ropFor(i) -= ropRev(i);
        ropFor(i) *= Crnd(i);
        for (ordinal_type j = 0; j < kmcd.reacNreac(i); ++j) {
          const ordinal_type kspec = kmcd.reacSidx(i, j);
          Kokkos::atomic_add(&omega(kspec), kmcd.reacNuki(i, j) * ropFor(i));
        }
        const ordinal_type joff = kmcd.reacSidx.extent(1) / 2;
        for (ordinal_type j = 0; j < kmcd.reacNprod(i); ++j) {
          const ordinal_type kspec = kmcd.reacSidx(i, j + joff);
          Kokkos::atomic_add(&omega(kspec), kmcd.reacNuki(i, j + joff) * ropFor(i));
        }
      });
#endif
    member.team_barrier();

    /// 9. transform from mole/(cm3.s) to kmol/(m3.s)
    {
      const real_type one_e_3(1e3);
      Kokkos::parallel_for(Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
                           [&](const ordinal_type& i) { omega(i) *= one_e_3; });
    }
#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("ReactionRatesSacado.team_invoke.test.out", "a+");
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

  template<typename MemberType>
  KOKKOS_FORCEINLINE_FUNCTION static void team_invoke_sacado(
    const MemberType& member,
    /// input
    const value_type& t,
    const value_type& p,
    const value_type& density,
    const value_type_1d_view_type& Ys, /// (kmcd.nSpec)
    /// output
    const value_type_1d_view_type& omega, /// (kmcd.nSpec)
    /// workspace
    const real_type_1d_view_type& work,
    /// kmcd.nSpec(4) : concX,gk,hks,cpks,
    /// kmcd.nReac(5) : concM,kfor,krev,rop,Crnd
    /// const input from kinetic model
    const kinetic_model_type& kmcd)
  {

    const ordinal_type sacadoStorageDimension = ats<value_type>::sacadoStorageDimension(t);
    auto w = (real_type*)work.data();
    const ordinal_type len = value_type().length();

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

    auto iter = ordinary_type_1d_view_type((ordinal_type*)w, kmcd.nReac * 2);
    w += kmcd.nReac * 2;

    const ordinal_type work_kfor_rev_size =
    Impl::KForwardReverse<value_type,device_type>::getWorkSpaceSize(kmcd);
    auto work_kfor_rev = value_type_1d_view_type(w, work_kfor_rev_size, sacadoStorageDimension);
    w += work_kfor_rev_size*len;

    team_invoke_detail(member,
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
  }

  template<typename MemberType>
  KOKKOS_FORCEINLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const real_type& t,
    const real_type& p,
    const real_type& density,
    const real_type_1d_view_type& Ys, /// (kmcd.nSpec)
    /// output
    const real_type_1d_view_type& omega, /// (kmcd.nSpec)
    /// workspace
    const real_type_1d_view_type& work,
    /// kmcd.nSpec(4) : concX,gk,hks,cpks,
    /// kmcd.nReac(5) : concM,kfor,krev,rop,Crnd
    /// const input from kinetic model
    const kinetic_model_type& kmcd)
  {
    ///const real_type zero(0);  /// not used

    ///
    /// workspace needed gk, hks, kfor, krev
    ///
    auto w = (real_type*)work.data();

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

    auto iter = ordinary_type_1d_view_type((ordinal_type*)w, kmcd.nReac * 2);
    w += kmcd.nReac * 2;

    const ordinal_type work_kfor_rev_size =
    Impl::KForwardReverse<real_type,device_type>::getWorkSpaceSize(kmcd);
    auto work_kfor_rev = real_type_1d_view_type(w, work_kfor_rev_size);
    w += work_kfor_rev_size;

    team_invoke_detail(member,
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
  }
};


} // namespace Impl
} // namespace TChem

#endif
