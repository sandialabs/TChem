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
#ifndef __TCHEM_IMPL_RATEOFPROGRESS_SURFACE_HPP__
#define __TCHEM_IMPL_RATEOFPROGRESS_SURFACE_HPP__

#include "TChem_Util.hpp"
#include "TChem_Impl_SurfaceCoverageModification.hpp"

namespace TChem {
namespace Impl {

///
/// ...
///
struct RateOfProgressSurface
{
  template<typename MemberType,
           typename RealType1DViewType,
           typename KineticSurfModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    const real_type &t,
    /// input
    const RealType1DViewType& kfor,
    const RealType1DViewType& krev,
    const RealType1DViewType& concX,
    const RealType1DViewType& concXSurf,
    /// output
    const RealType1DViewType& ropFor,
    const RealType1DViewType& ropRev,
    // work
    const RealType1DViewType& CoverageFactor,
    /// const input from kinetic model
    const KineticSurfModelConstDataType& kmcdSurf)
  {

    SurfaceCoverageModification::team_invoke(member,
    t, concX, concXSurf, CoverageFactor, kmcdSurf);

    member.team_barrier();

    Kokkos::parallel_for(
      Kokkos::TeamVectorRange(member, kmcdSurf.nReac),
      [&](const ordinal_type& i) {
        real_type ropFor_at_i = kfor(i)*CoverageFactor(i);
        real_type ropRev_at_i = krev(i)*CoverageFactor(i);

        /* compute forward rop */
        for (ordinal_type j = 0; j < kmcdSurf.reacNreac(i); ++j) {
          const ordinal_type kspec = kmcdSurf.reacSidx(i, j); // species index
          const ordinal_type niup =
            ats<ordinal_type>::abs(kmcdSurf.reacNuki(i, j)); // st coef
          if (kmcdSurf.reacSsrf(i, j) == 1) {                // surface
            ropFor_at_i *= ats<real_type>::pow(concXSurf(kspec), niup);
          } else { // gas
            ropFor_at_i *= ats<real_type>::pow(concX(kspec), niup);
          }
        }

        if (kmcdSurf.isRev(i)) {
          /* compute reverse rop */
          for (ordinal_type j = 0; j < kmcdSurf.reacNprod(i); j++) {

            const ordinal_type joff = kmcdSurf.maxSpecInReac / 2;
            const ordinal_type kspec =
              kmcdSurf.reacSidx(i, j + joff); // species index
            const ordinal_type nius =
              ats<ordinal_type>::abs(kmcdSurf.reacNuki(i, j + joff)); // st coef

            if (kmcdSurf.reacSsrf(i, j + joff) == 1) { // surface
              ropRev_at_i *= ats<real_type>::pow(concXSurf(kspec), nius);
            } else { // gas
              ropRev_at_i *= ats<real_type>::pow(concX(kspec), nius);
            }
          }
        }
        ropFor(i) = ropFor_at_i;
        ropRev(i) = ropRev_at_i;
      }); /* done loop over all reactions */

#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("RateOfProgressSurface.team_invoke.test.out", "a+");
      fprintf(fs, ":: RateOfProgressSurface::team_invoke\n");
      fprintf(fs, ":::: input\n");
      fprintf(fs,
              "     nSpec %3d, nReac %3d, kfor %3d, krev %3d, concX %3d\n",
              kmcdSurf.nSpec,
              kmcdSurf.nReac,
              int(kfor.extent(0)),
              int(krev.extent(0)),
              int(concX.extent(0)));
      for (int i = 0; i < int(kfor.extent(0)); ++i)
        fprintf(fs, "     i %3d, kfor %e, krev %e\n", i, kfor(i), krev(i));
      for (int i = 0; i < int(concX.extent(0)); ++i)
        fprintf(fs, "     i %3d, concX %e\n", i, concX(i));
      fprintf(fs, ":::: output\n");
      // fprintf(fs,   "     ropFor %3d, ropRev %3d\n",
      //        int(ropFor.extent(0)), int(ropRev.extent(0)));
      for (int i = 0; i < int(ropFor.extent(0)); ++i)
        fprintf(
          fs, "     i %3d, ropFor %e, ropRev %e\n", i, ropFor(i), ropRev(i));
    }
#endif
  }

  template<typename MemberType,
           typename WorkViewType,
           typename RealType1DViewType,
           typename KineticSurfModelConstDataType>
  KOKKOS_FORCEINLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    const real_type &t,
    /// input
    const RealType1DViewType& kfor,
    const RealType1DViewType& krev,
    const RealType1DViewType& concX,
    const RealType1DViewType& concXSurf,
    /// output
    const RealType1DViewType& ropFor,
    const RealType1DViewType& ropRev,
    /// work
    const WorkViewType& work,
    /// const input from kinetic model
    const KineticSurfModelConstDataType& kmcd)
  {

    auto w = (real_type*)work.data();
    auto CoverageFactor = RealType1DViewType(w, kmcd.nReac);
    w += kmcd.nReac;

    team_invoke_detail(
      member, t,  kfor, krev, concX, concXSurf, ropFor, ropRev, CoverageFactor, kmcd);
  }
};

} // namespace Impl
} // namespace TChem

#endif
