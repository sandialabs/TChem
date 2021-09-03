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
#ifndef __TCHEM_IMPL_REACTIONRATESTOYPROBLEM_HPP__
#define __TCHEM_IMPL_REACTIONRATESTOYPROBLEM_HPP__

#include "TChem_Impl_RateOfProgress.hpp"
#include "TChem_Util.hpp"
// #define TCHEM_ENABLE_SERIAL_TEST_OUTPUT
namespace TChem {
namespace Impl {

template<typename ValueType, typename DeviceType>
struct SourceTermToyProblem
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
    return 6 * kmcd.nReac;
  }

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    /// input
    const real_type& theta,
    const real_type& lambda,
    const real_type_1d_view_type& concX,
    /// output
    const real_type_1d_view_type& omega, /// (kmcd.nSpec)
    const real_type_1d_view_type& kfor,
    const real_type_1d_view_type& krev,
    const real_type_1d_view_type& ropFor,
    const real_type_1d_view_type& ropRev,
    const ordinary_type_1d_view_type& iter,
    /// const input from kinetic model
    const kinetic_model_type& kmcd)
  {
    // const real_type zero(0);
    using RateOfProgress = RateOfProgress<real_type,device_type>;


    /// 1. compute forward and reverse rate constants
    {
      const real_type thetac(20.0); //! degrees
      const real_type lambdac(300.0);// ! degrees
      const real_type k1 = ats<real_type>::sin(theta*PI()/180) * ats<real_type>::sin(thetac*PI()/180) +
                           ats<real_type>::cos(theta*PI()/180) * ats<real_type>::cos(thetac*PI()/180) *
                           ats<real_type>::cos(lambda*PI()/180 - lambdac*PI()/180);
      kfor(0) = k1 > 0 ? k1 : 0;
      kfor(1) = 1;
      krev(0) = 0;
      krev(0) = 0;

      // printf("k1 = %e\n",kfor(0) );
      // printf("k2 = %e\n",kfor(1) );
    }

    member.team_barrier();


    /// 2. compute rate-of-progress
    RateOfProgress::team_invoke(member,
                                kfor,
                                krev,
                                concX, /// input
                                ropFor,
                                ropRev, /// output
                                iter,   /// workspace for iterators
                                kmcd);

    member.team_barrier();

    /// 3. assemble reaction rates
    auto rop = ropFor;
    Kokkos::parallel_for(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nReac), [&](const ordinal_type& i) {
        rop(i) -= ropRev(i);
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


    member.team_barrier();

#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("SourceTermToyProblem.team_invoke.test.out", "a+");
      fprintf(fs, ":: SourceTermToyProblem::team_invoke\n");
      fprintf(fs, ":::: input\n");
      fprintf(fs,
              "     nSpec %3d, nReac %3d\n",
              kmcd.nSpec,
              kmcd.nReac);
      //
      fprintf(fs, ":: concX\n");
      for (int i = 0; i < int(concX.extent(0)); ++i)
        fprintf(fs,
                "     i %3d, kfor %e\n",
                i,
                concX(i));
      fprintf(fs, ":: kfor\n");
      for (int i = 0; i < int(kfor.extent(0)); ++i)
        fprintf(fs,
                "     i %3d, kfor %e\n",
                i,
                kfor(i));
      fprintf(fs, ":: krev\n");
      for (int i = 0; i < int(krev.extent(0)); ++i)
        fprintf(fs,
                "     i %3d, krev %e\n",
                i,
                krev(i));
      //
      fprintf(fs, ":: ropFor\n");
      for (int i = 0; i < int(ropFor.extent(0)); ++i)
        fprintf(fs,
                "     i %3d, kfor %e\n",
                i,
                ropFor(i));
      fprintf(fs, ":: ropRev\n");
      for (int i = 0; i < int(ropRev.extent(0)); ++i)
        fprintf(fs,
                "     i %3d, krev %e\n",
                i,
                ropRev(i));
      fprintf(fs, "\n");
      fprintf(fs, ":::: output\n");
      for (int i = 0; i < int(omega.extent(0)); ++i)
        fprintf(fs, "     i %3d, omega %e\n", i, omega(i));
      fprintf(fs, "\n");
    }
#endif

    // if (member.league_rank() == 0) {
    //   FILE *fs = fopen("SourceTermToyProblem.team_invoke.test.out", "a+");
    //   for (int i=0;i<int(Crnd.extent(0));++i)
    //     fprintf(fs, " %d %e\n", i , real_type(1e-3)*Crnd(i)*kfor(i));
    // }
  }


  template<typename MemberType,
           typename WorkViewType>
  KOKKOS_FORCEINLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const real_type& theta,
    const real_type& lambda,
    const real_type_1d_view_type& X, /// (kmcd.nSpec)
    /// output
    const real_type_1d_view_type& omega, /// (kmcd.nSpec)
    /// workspace
    const WorkViewType& work,
    /// const input from kinetic model
    const kinetic_model_type& kmcd)
  {

    ///
    auto w = (real_type*)work.data();
    auto kfor = real_type_1d_view_type(w, kmcd.nReac);
    w += kmcd.nReac;
    auto krev = real_type_1d_view_type(w, kmcd.nReac);
    w += kmcd.nReac;
    auto ropFor = real_type_1d_view_type(w, kmcd.nReac);
    w += kmcd.nReac;
    auto ropRev = real_type_1d_view_type(w, kmcd.nReac);
    w += kmcd.nReac;
    auto iter = ordinary_type_1d_view_type((ordinal_type*)w, kmcd.nReac * 2);

    w += kmcd.nReac * 2;

    team_invoke_detail(member,
                       theta,
                       lambda,
                       X,
                       omega,
                       kfor,
                       krev,
                       ropFor,
                       ropRev,
                       iter,
                       kmcd);
  }

};

} // namespace Impl
} // namespace TChem

#endif
