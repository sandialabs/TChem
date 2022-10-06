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
#ifndef __TCHEM_IMPL_KFORWARDREVERSE_SURFACE_HPP__
#define __TCHEM_IMPL_KFORWARDREVERSE_SURFACE_HPP__

#include "TChem_Impl_SumNuGk.hpp"
#include "TChem_Util.hpp"

namespace TChem {
namespace Impl {

template<typename ValueType, typename DeviceType>
struct KForwardReverseSurface
{
  using value_type = ValueType;
  using device_type = DeviceType;
  using scalar_type = typename ats<value_type>::scalar_type;

  using real_type = scalar_type;
  /// sacado is value type
  using value_type_1d_view_type = Tines::value_type_1d_view<value_type,device_type>;
  using kinetic_model_type= KineticModelConstData<device_type>;
  using kinetic_surf_model_type = KineticSurfModelConstData<device_type>;

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input temperature
    const value_type& t,
    const value_type& p,
    const value_type_1d_view_type& gk,
    const value_type_1d_view_type& gkSurf,
    /// output
    const value_type_1d_view_type& kfor,
    const value_type_1d_view_type& krev,
    /// const input from kinetic model
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf)
  {

    using SumNuGk = SumNuGk<value_type, device_type>;

    const value_type t_1 = real_type(1) / t;
    const value_type tln = ats<value_type>::log(t);
    const real_type ten(10);
    // ordinal_type indx(0);

    value_type Wk(0);

    Kokkos::parallel_for(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcdSurf.nReac),
      [&](const ordinal_type& i) {
        kfor(i) = (kmcdSurf.reacArhenFor(i, 0) *
                   ats<value_type>::exp(kmcdSurf.reacArhenFor(i, 1) * tln -
                                       kmcdSurf.reacArhenFor(i, 2) * t_1));

        if (kmcdSurf.isStick(i) == 1) {
          value_type gamma = kfor(i);
          // eq 16.117 chapter 16 Robert J. Kee
          ordinal_type m(0);
          for (ordinal_type j = 0; j < kmcdSurf.reacNreac(i); j++) {
            if (kmcdSurf.reacSsrf(i, j) == 1) { // if species is surface
              m += ats<ordinal_type>::abs(kmcdSurf.reacNuki(i, j));
            } else if (kmcdSurf.reacSsrf(i, j) == 0) { // if species is gas
              // get Wk
              Wk = kmcd.sMass(kmcdSurf.reacSidx(
                i, j)); // check this, assume only one species?
            }
          }
          // Motz–Wise Correction eq 16.118 chapter 16 Robert J. Kee
          if (kmcdSurf.motz_wise) {
            gamma /= (real_type(1) - real_type(0.5)*gamma);
          }

          kfor(i) = gamma * ats<value_type>::sqrt(kmcd.Rcgs * t / (DPI() * Wk)) /
                     ats<value_type>::pow(kmcdSurf.sitedensity, m);


        }

        /* is reaction reversible ? */

        ordinal_type nusum = 0;
        ordinal_type nusum2 = 0;
        if (kmcdSurf.isRev(i)) {
          // eq 16.105 chapter 16 Robert J. Kee
          /* no, need to compute equilibrium constant */
          for (ordinal_type j = 0; j < kmcdSurf.reacNreac(i); j++) {
            if (kmcdSurf.reacSsrf(i, j) == 0) { // if species is gas
              nusum += kmcdSurf.reacNuki(i, j);
            } else if (kmcdSurf.reacSsrf(i, j) == 1) {
              nusum2 += kmcdSurf.reacNuki(i, j);
            }
          }
          const ordinal_type joff = kmcdSurf.maxSpecInReac / 2;

          for (ordinal_type j = 0; j < kmcdSurf.reacNprod(i); j++) {
            if (kmcdSurf.reacSsrf(i, j + joff) == 0) { // gas
              nusum += kmcdSurf.reacNuki(i, j + joff);
            } else if (kmcdSurf.reacSsrf(i, j + joff) == 1) {
              nusum2 += kmcdSurf.reacNuki(i, j + joff);
            }
          }

          const value_type sumNuGk =
            SumNuGk::serial_invoke(i, gk, gkSurf, kmcdSurf);

          const value_type kc =
            ats<value_type>::pow((ATMPA() * ten / kmcd.Rcgs) * t_1, nusum) *
            ats<value_type>::pow(kmcdSurf.sitedensity, nusum2) *
            ats<value_type>::exp(sumNuGk);

          krev(i) = kfor(i) / kc;

        } else {
          krev(i) = 0;
        } /* done if reaction is reversible */
      }); /* done computing kforward and kreverse rate constants */

#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("KForwardReverseSurface.team_invoke.test.out", "a+");
      fprintf(fs, ":: KForwardReverseSurface::team_invoke\n");
      fprintf(fs, ":::: input\n");
      fprintf(fs,
              "     nSpec %3d, nReac %3d, t %e, p %e, site density %e\n",
              kmcdSurf.nSpec,
              kmcdSurf.nReac,
              t,
              p,
              kmcdSurf.sitedensity);
      for (int i = 0; i < kmcdSurf.nReac; ++i)
        fprintf(fs,
                "   i %3d,  A1 %e, A2 %e, A3 %e \n",
                i,
                kmcdSurf.reacArhenFor(i, 0),
                kmcdSurf.reacArhenFor(i, 1),
                kmcdSurf.reacArhenFor(i, 2));
      fprintf(fs, ":::: output\n");
      for (int i = 0; i < int(kfor.extent(0)); ++i)
        fprintf(fs, "     i %3d, kfor %e, krev %e\n", i, kfor(i), krev(i));
    }
#endif
  }

};

} // namespace Impl
} // namespace TChem

#endif
