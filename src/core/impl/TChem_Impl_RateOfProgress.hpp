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
#ifndef __TCHEM_IMPL_RATEOFPROGRESS_HPP__
#define __TCHEM_IMPL_RATEOFPROGRESS_HPP__

#include "TChem_Util.hpp"
// #define TCHEM_ENABLE_SERIAL_TEST_OUTPUT
namespace TChem {
namespace Impl {

///
/// ...
///
struct RateOfProgress
{
  template<typename MemberType,
           typename RealType1DViewType,
           typename OrdinalType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    /// input
    const RealType1DViewType& kfor,
    const RealType1DViewType& krev,
    const RealType1DViewType& concX,
    /// output
    const RealType1DViewType& ropFor,
    const RealType1DViewType& ropRev,
    /// work
    const OrdinalType1DViewType& irnus,
    const OrdinalType1DViewType& iords,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    Kokkos::single(Kokkos::PerTeam(member), [&]() {
      /// compute iterators
      ordinal_type irnu(0), iord(0);
      for (ordinal_type i = 0; i < kmcd.nReac; ++i) {
        /// we compute only one case
        const bool iord_flag = iord < kmcd.nOrdReac && kmcd.reacAOrd(iord) == i;

        /// store iterators
        irnus(i) = irnu;
        iords(i) = iord;

        /// increase iterators
        irnu += !iord_flag;
        iord += iord_flag;
      }
    });
    member.team_barrier();

    Kokkos::parallel_for(
      Kokkos::TeamVectorRange(member, kmcd.nReac), [&](const ordinal_type& i) {
        real_type ropFor_at_i = kfor(i);
        real_type ropRev_at_i = krev(i);

        /// we compute only one case
        const ordinal_type irnu = irnus(i), iord = iords(i);

        const bool iord_flag = iord < kmcd.nOrdReac && kmcd.reacAOrd(iord) == i;

        if (!iord_flag) {
          /* compute forward rop */
          for (ordinal_type j = 0; j < kmcd.reacNreac(i); ++j) {
            const ordinal_type kspec = kmcd.reacSidx(i, j);
            const real_type niup =
              ats<real_type>::abs(kmcd.reacNuki(i, j));
            ropFor_at_i *= ats<real_type>::pow(concX(kspec), niup);
          }

          if (kmcd.isRev(i)) {
            /* compute reverse rop */
            const ordinal_type joff = kmcd.reacSidx.extent(1) / 2;
            for (ordinal_type j = 0; j < kmcd.reacNprod(i); ++j) {
              const ordinal_type kspec = kmcd.reacSidx(i, j + joff);
              const real_type nius = kmcd.reacNuki(i, j + joff);
              ropRev_at_i *= ats<real_type>::pow(concX(kspec), nius);
            }
          }
        }

        /* check for arbitrary order reaction */
        if (iord_flag) {
          for (ordinal_type j = 0; j < kmcd.maxOrdPar; ++j) {
            const ordinal_type kspec =
              ats<ordinal_type>::abs(kmcd.specAOidx(i, j)) - 1;
            const real_type niu = kmcd.specAOval(i, j);
#ifdef NONNEG
            const real_type concX_value_at_kspec =
              ats<real_type>::abs(concX(kspec));
#else
                const real_type concX_value_at_kspec = concX(kspec);
#endif
            if (kmcd.specAOidx(i, j) < 0) {
              const real_type niup = niu;
              ropFor_at_i *= ats<real_type>::pow(concX_value_at_kspec, niup);
            } else if (kmcd.specAOidx(i, j) > 0) {
              const real_type nius = niu;
              ropRev_at_i *= ats<real_type>::pow(concX_value_at_kspec, nius);
            }
          } /* done if arbitrary order reaction */
        }
        ropFor(i) = ropFor_at_i;
        ropRev(i) = ropRev_at_i;
      }); /* done loop over all reactions */

#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("RateOfProgress.team_invoke.test.out", "a+");
      fprintf(fs, ":: RateOfProgress::team_invoke\n");
      fprintf(fs, ":::: input\n");
      fprintf(fs,
              "     nSpec %3d, nReac %3d, kfor %3d, krev %3d, concX %3d\n",
              kmcd.nSpec,
              kmcd.nReac,
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
           typename KineticModelConstDataType>
  KOKKOS_FORCEINLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const RealType1DViewType& kfor,
    const RealType1DViewType& krev,
    const RealType1DViewType& concX,
    /// output
    const RealType1DViewType& ropFor,
    const RealType1DViewType& ropRev,
    /// work
    const WorkViewType& work,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    auto w = (ordinal_type*)work.data();
    auto irnus =
      Kokkos::View<ordinal_type*,
                   Kokkos::LayoutRight,
                   typename WorkViewType::memory_space>(w, kmcd.nReac);
    w += kmcd.nReac;
    auto iords =
      Kokkos::View<ordinal_type*,
                   Kokkos::LayoutRight,
                   typename WorkViewType::memory_space>(w, kmcd.nReac);
    w += kmcd.nReac;

    team_invoke_detail(
      member, kfor, krev, concX, ropFor, ropRev, irnus, iords, kmcd);
  }
};

template<typename ControlType>
KOKKOS_INLINE_FUNCTION bool
RateOfProgressDerivativeEnableRealStoichiometricCoefficients(
  const ControlType& control)
{
  return true;
}

template<typename ControlType>
KOKKOS_INLINE_FUNCTION bool
RateOfProgressDerivativeEnableArbitraryOrderTerms(const ControlType& control)
{
  return true;
}

struct RateOfProgressDerivative
{
  template<typename ControlType,
           typename RealType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void serial_invoke(
    const ControlType& control,
    /// input
    const ordinal_type ir, /// reaction idx
    const ordinal_type is, /// species idx
    const RealType1DViewType& kfor,
    const RealType1DViewType& krev,
    const RealType1DViewType& concX,
    /// output
    /* */ real_type& qfor,
    /* */ real_type& qrev,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    const real_type zero(0), one(1);

    /// check real stoichiometric coefficients
    /// with stoichiometric coefficients, it compute qfor/qrev from scratch
    ordinal_type ir2irnu = -1;
    for (ordinal_type irnu = 0; irnu < kmcd.nRealNuReac; ++irnu)
      if (kmcd.reacRnu(irnu) == ir) {
        ir2irnu = irnu;
        break;
      }

    /// for arbitrary order terms, it recomputes
    ordinal_type ir2iord = -1;
    for (ordinal_type iord = 0; iord < kmcd.nOrdReac; ++iord)
      if (kmcd.reacAOrd(iord) == ir) {
        ir2iord = iord;
        break;
      }

    const bool irnu_flag = ir2irnu > 0;
    const bool iord_flag = ir2iord > 0;

    /// minus input of qfor and qrev indicates compute
    const bool compute_qfor = qfor < zero;
    const bool compute_qrev = qrev < zero;

    qfor = kfor(ir);
    qrev = zero;

    if (!irnu_flag && !iord_flag) {
      /// This part is tested
      /// compute forward qfor
      if (compute_qfor) {
        for (ordinal_type j = 0; j < kmcd.reacNreac(ir); ++j) {
          const ordinal_type kspec = kmcd.reacSidx(ir, j);
          const real_type niup =
            ats<real_type>::abs(kmcd.reacNuki(ir, j));
          if (is == kspec) {
            qfor *=
              real_type(niup) * ats<real_type>::pow(concX(kspec), niup - 1);
          } else {
            qfor *= ats<real_type>::pow(concX(kspec), niup);
          }
        } /* done loop over reactants */
      }
      if (compute_qrev) {
        if (kmcd.isRev(ir)) {
          /* compute reverse qrev */
          qrev = krev(ir);
          const ordinal_type joff = kmcd.reacSidx.extent(1) / 2;
          for (ordinal_type j = 0; j < kmcd.reacNprod(ir); ++j) {
            const ordinal_type kspec = kmcd.reacSidx(ir, joff + j);
            const real_type nius = kmcd.reacNuki(ir, joff + j);
            if (is == kspec) {
              qrev *=
                real_type(nius) * ats<real_type>::pow(concX(kspec), nius - 1);
            } else {
              qrev *= ats<real_type>::pow(concX(kspec), nius);
            }
          }
        }
      }
    }

    if (RateOfProgressDerivativeEnableRealStoichiometricCoefficients(control)) {
      if (irnu_flag && !iord_flag) {
        /// This is not tested
        /* Check for real stoichiometric coefficients */
        if (compute_qfor) {
          const ordinal_type irnu = ir2irnu;
          for (ordinal_type j = 0; j < kmcd.reacNreac(ir); ++j) {
            const ordinal_type kspec = kmcd.reacSidx(ir, j);
            const real_type niup =
              kmcd.reacRealNuki.span() > 0
                ? ats<real_type>::abs(kmcd.reacRealNuki(irnu, j))
                : zero;
            const real_type concX_at_kspec =
              concX(kspec) > zero ? concX(kspec) : 1.e-20;
            if (is == kspec)
              qfor *= niup * ats<real_type>::pow(concX_at_kspec, niup - 1);
            else
              qfor *= ats<real_type>::pow(concX_at_kspec, niup);
          } /* Done loop over all reactants */
        }
        if (compute_qrev) {
          if (kmcd.isRev(ir)) {
            /* compute reverse qrev */
            qrev = krev(ir);

            const ordinal_type joff = kmcd.reacSidx.extent(1) / 2;
            for (ordinal_type j = 0; j < kmcd.reacNprod(ir); ++j) {
              const ordinal_type kspec = kmcd.reacSidx(ir, j + joff);
              const real_type nius = kmcd.reacRealNuki.span() > 0
                                       ? kmcd.reacRealNuki(ir, j + joff)
                                       : zero;
              const real_type concX_at_kspec =
                concX(kspec) > zero ? concX(kspec) : 1.e-20;
              if (is == kspec)
                qrev *= nius * ats<real_type>::pow(concX_at_kspec, nius - 1);
              else
                qrev *= ats<real_type>::pow(concX_at_kspec, nius);
            } /* Done loop over all products */
          }   /* Done if real coef reac is rev */
        }
      }
    }

    /* check for arbitrary order reaction */
    if (RateOfProgressDerivativeEnableArbitraryOrderTerms(control)) {
      if (iord_flag) {
        /// This is not tested
        if (compute_qfor) {
          const ordinal_type iord = ir2iord;
          qfor = kfor(ir);
          for (ordinal_type j = 0; j < kmcd.maxOrdPar; ++j) {
            const ordinal_type kspec =
              ats<ordinal_type>::abs(kmcd.specAOidx(iord, j)) - 1;
            const real_type spec_ao_val = kmcd.specAOval(iord, j);
            const real_type concX_at_kspec =
              concX(kspec) > zero ? concX(kspec) : real_type(1.e-20);
            if (is == kspec) {
              qfor *= spec_ao_val *
                      ats<real_type>::pow(concX_at_kspec, spec_ao_val - one);
            } else {
              qfor *= ats<real_type>::pow(concX_at_kspec, spec_ao_val);
            }
          }
        }
        if (compute_qrev)
          qrev = krev(ir);
      } /* done if arbitrary order reaction */
    }
#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    /// serial interface cannot export verbose output as
    /// it has to assume team collaborative environment.
    // {
    //   FILE *fs = fopen("RateOfProgressDerivative.team_invoke.test.out",
    //   "a+"); fprintf(fs,   ":: RateOfProgressDerivative::team_invoke\n");
    //   fprintf(fs,   ":::: input\n");
    //   fprintf(fs,   "     nReac %3d, ir %3d, is %3d\n", kmcd.nReac, ir, is);
    //   for (int i=0;i<int(kfor.extent(0));++i)
    //     fprintf(fs, "     i %3d, kfor %e, krev %e\n", i, kfor(i), krev(i));
    //   for (int i=0;i<int(concX.extent(0));++i)
    //     fprintf(fs, "     i %3d, concX %e\n", i, concX(i));
    //   fprintf(fs,   ":::: output\n");
    //   fprintf(fs,   "     qfor %e, qrev %e\n", qfor, qrev);
    // }
#endif
  }
};

} // namespace Impl
} // namespace TChem

#endif
