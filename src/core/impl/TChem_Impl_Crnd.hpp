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
#ifndef __TCHEM_IMPL_CRND_HPP__
#define __TCHEM_IMPL_CRND_HPP__

#include "TChem_Util.hpp"
#include "TChem_KineticModelData.hpp"

namespace TChem {
namespace Impl {

///
///
///
template<typename ValueType, typename DeviceType>
struct Crnd
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

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    /// input
    const value_type& t,
    const value_type_1d_view_type& kfor,
    const value_type_1d_view_type& concX,
    const value_type_1d_view_type& concM,
    /// output
    const value_type_1d_view_type& Crnd,
    /// work
    const ordinary_type_1d_view_type& ipfals,
    /// const input from kinetic model
    const kinetic_model_type& kmcd)
  {
    const real_type one(1), zero(0);
    const value_type t_1 = one / t;
    const value_type tln = ats<value_type>::log(t);

    Kokkos::single(Kokkos::PerTeam(member), [&]() {
      /// compute iterators
      ordinal_type ipfal(0);
      for (ordinal_type i = 0; i < kmcd.nReac; ++i) {
        ipfals(i) = ipfal;
        ipfal += (ipfal < kmcd.nFallReac) && (kmcd.reacPfal(ipfal) == i);
      }
    });
    member.team_barrier();

    Kokkos::parallel_for(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nReac), [&](const ordinal_type& i) {
        Crnd(i) = concM(i);
        const ordinal_type ipfal = ipfals(i);
        if (ipfal < kmcd.nFallReac) {
          if (kmcd.reacPfal(ipfal) == i) {

            value_type Pr(0);
            auto rp = Kokkos::subview(kmcd.reacPpar, ipfal, Kokkos::ALL());

            if (kmcd.reacPlohi(ipfal) == 0) {
              /* LOW reaction */
              const value_type k0 =
                rp(0) * ats<value_type>::exp(rp(1) * tln - rp(2) * t_1);
              Pr = k0 / kfor(i);
            } else {
              /* HIGH reaction */
              const value_type kinf =
                rp(0) * ats<value_type>::exp(rp(1) * tln - rp(2) * t_1);
              Pr = kfor(i) / kinf;
            }
            Pr *= (kmcd.reacPspec(ipfal) >= 0 ? concX(kmcd.reacPspec(ipfal))
                                              : concM(i));
            Crnd(i) = Pr / (one + Pr); /* At least Lindemann form */

            const value_type logPr =
              ats<value_type>::log10(Pr > TCSMALL() ? Pr : TCSMALL());

            /// SRI form
            if (kmcd.reacPtype(ipfal) == 2) {
              const value_type Xpres = one / (one + logPr * logPr);
              const value_type Ffac =
                ats<value_type>::pow(rp(3) * ats<value_type>::exp(-rp(4) * t_1) +
                                      ats<value_type>::exp(-t / rp(5)),
                                    Xpres) *
                rp(6) * ats<value_type>::pow(t, rp(7));
              Crnd(i) *= Ffac;
            } /* done with SRI form */
            /// TROE form
            else if (kmcd.reacPtype(ipfal) >= 3) {
              // real_type Fc(0);
              // if (ats<real_type>::abs(one-rp(6)) > zero) Fc +=
              // (one-rp(3))*exp(-t/rp(4)); if (ats<real_type>::abs(    rp(6)) >
              // zero) Fc +=      rp(3) *exp(-t/rp(5)) ;

              const value_type Fc =
                ((1.0 - rp(3)) * ats<value_type>::exp(-t / rp(4)) +
                 rp(3) * ats<value_type>::exp(-t / rp(5)) +
                 (kmcd.reacPtype(ipfal) == 4 ? ats<value_type>::exp(-rp(6) * t_1)
                                             : zero));

              const value_type logFc = ats<value_type>::log10(Fc);
              const value_type Atroe = logPr - 0.40 - 0.67 * logFc;
              const value_type Btroe = 0.75 - 1.27 * logFc - 0.14 * Atroe;
              const value_type Atroe_Btroe = Atroe / Btroe;
              const value_type logFfac =
                logFc / (one + Atroe_Btroe * Atroe_Btroe);
              const value_type Ffac =
                ats<value_type>::pow(real_type(10), logFfac);

              Crnd(i) *= Ffac;
            } /* done with Troe form */
          }   /* done if pressure dependent reaction */
        }
      });

#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("Crnd.team_invoke.test.out", "a+");
      fprintf(fs, ":: Crnd::team_invoke\n");
      fprintf(fs, ":::: input\n");
      fprintf(fs, "     nSpec %3d, nReac %3d\n", kmcd.nSpec, kmcd.nReac);
      for (int i = 0; i < int(kfor.extent(0)); ++i)
        fprintf(fs, "     i %3d, kfor %e, concM %e\n", i, kfor(i), concM(i));
      for (int i = 0; i < int(concX.extent(0)); ++i)
        fprintf(fs, "     i %3d, concX %e\n", i, concX(i));
      fprintf(fs, ":::: output\n");
      for (int i = 0; i < int(Crnd.extent(0)); ++i)
        fprintf(fs, "     i %3d, Crnd %e\n", i, Crnd(i));
    }
#endif
  }

  template<typename MemberType,
           typename WorkViewType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const value_type& t,
    const value_type_1d_view_type& kfor,
    const value_type_1d_view_type& concX,
    const value_type_1d_view_type& concM,
    /// output
    const value_type_1d_view_type& Crnd,
    /// work
    const WorkViewType& work,
    /// const input from kinetic model
    const kinetic_model_type& kmcd)
  {
    auto w = (ordinal_type*)work.data();
    // auto ipfals =
    //   Kokkos::View<ordinal_type*,
    //                Kokkos::LayoutRight,
    //                typename WorkViewType::memory_space>(w, kmcd.nReac);
    auto ipfals =ordinary_type_1d_view_type(w, kmcd.nReac);
    w += kmcd.nReac;

    team_invoke_detail(member, t, kfor, concX, concM, Crnd, ipfals, kmcd);
  }
};

struct CrndDerivative
{
  template<typename MemberType,
           typename WorkViewType,
           typename RealType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const ordinal_type& ireac,
    const real_type& t,
    const RealType1DViewType& kfor,
    const RealType1DViewType& concX,
    const RealType1DViewType& concM,
    /// input/output
    /* */ ordinal_type& itbdy,
    /* */ ordinal_type& ipfal,
    /// output (t, nSpec)
    /* */ real_type& CrndDt,
    const RealType1DViewType& CrndDer,
    const WorkViewType& work,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    typedef Kokkos::pair<ordinal_type, ordinal_type> range_type;

    const real_type two(2), one(1), zero(0);
    const real_type t_1 = one / t;
    const real_type tln = ats<real_type>::log(t);

#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    const ordinal_type itbdy_prev = itbdy;
    const ordinal_type ipfal_prev = ipfal;
#endif
    /// initialize
    CrndDt = zero;
    const auto PrDer = work;
    real_type PrDt(0);
    {
      Kokkos::parallel_for(Tines::RangeFactory<real_type>::TeamVectorRange(member, kmcd.nSpec),
                           [&](const ordinal_type& i) {
                             CrndDer(i) = zero;
                             PrDer(i) = zero;
                           });
    }
    member.team_barrier();

    ///
    /// Third-body reactions
    ///
    if (itbdy < kmcd.nThbReac) {
      if (kmcd.reacTbdy(itbdy) == ireac) {

        Kokkos::parallel_for(Tines::RangeFactory<real_type>::TeamVectorRange(member, kmcd.nSpec),
                             [&](const ordinal_type& i) { CrndDer(i) = one; });
        member.team_barrier();
        Kokkos::parallel_for(
          Tines::RangeFactory<real_type>::TeamVectorRange(member, kmcd.reacTbno(itbdy)),
          [&](const ordinal_type& j) {
            const ordinal_type kspec = kmcd.specTbdIdx(itbdy, j);
            CrndDer(kspec) = kmcd.specTbdEff(itbdy, j);
          }); /// done loop over efficiences
        ++itbdy;
      } /// if reaction involves third-body
    }   /// done if there are third-body reaction left

    ///
    /// Pressure dependent reactions
    ///
    if (ipfal < kmcd.nFallReac) {
      if (kmcd.reacPfal(ipfal) == ireac) {

        /// Compute Pr and its derivatives
        real_type Pr(0);
        auto rp = Kokkos::subview(kmcd.reacPpar, ipfal, Kokkos::ALL());
        auto ra = Kokkos::subview(kmcd.reacArhenFor, ireac, Kokkos::ALL());

        if (kmcd.reacPlohi(ipfal) == 0) {
          /* LOW reaction */
          const real_type k0 =
            rp(0) * ats<real_type>::exp(rp(1) * tln - rp(2) * t_1);
          Pr = k0 / kfor(ireac);
        } else {
          /* HIGH reaction */
          const real_type kinf =
            rp(0) * ats<real_type>::exp(rp(1) * tln - rp(2) * t_1);
          Pr = kfor(ireac) / kinf;
        }

        if (kmcd.reacPspec(ipfal) >= 0) {
          PrDer(kmcd.reacPspec(ipfal)) = Pr;
          Pr *= concX(kmcd.reacPspec(ipfal));
        } else {
          Kokkos::parallel_for(
            Tines::RangeFactory<real_type>::TeamVectorRange(member, kmcd.nSpec),
            [&](const ordinal_type& k) { PrDer(k) = Pr * CrndDer(k); });
          Pr *= concM(ireac);
        }

        if (kmcd.reacPlohi(ipfal) == 0) {
          /// LOW reaction
          PrDt = Pr * t_1 * (rp(1) - ra(1) + t_1 * (rp(2) - ra(2)));
        } else {
          /// HIGH reaction
          PrDt = Pr * t_1 * (ra(1) - rp(1) + t_1 * (ra(2) - rp(2)));
        }

        /// Lindemann form
        if (kmcd.reacPtype(ipfal) == 1) {
          const real_type Prfac = one / ((one + Pr) * (one + Pr));

          CrndDt = Prfac * PrDt;
          Kokkos::parallel_for(
            Tines::RangeFactory<real_type>::TeamVectorRange(member, kmcd.nSpec),
            [&](const ordinal_type& k) { CrndDer(k) = Prfac * PrDer(k); });
        }
        /// SRI form
        else if (kmcd.reacPtype(ipfal) == 2) {
          auto Psri = Kokkos::subview(rp, range_type(3, rp.extent(0)));
          const real_type logPr =
            ats<real_type>::log(Pr) / ats<real_type>::log(10);
          const real_type Xp = one / (one + logPr * logPr);
          const real_type dXp =
            -Xp * Xp * two * logPr / (Pr * ats<real_type>::log(10));
          const real_type abc = Psri(0) * ats<real_type>::exp(-Psri(1) * t_1) +
                                ats<real_type>::exp(-t / Psri(2));
          const real_type Ffac = (ats<real_type>::pow(abc, Xp) * Psri(3) *
                                  ats<real_type>::pow(t, Psri(4))) /
                                 ((one + Pr) * (one + Pr));
          const real_type Prfac = Pr / (one + Pr);
          const real_type abcS = ats<real_type>::log(abc) * dXp;

          {
            const real_type FfacDt =
              Ffac * (Psri(4) * t_1 + dXp * PrDt * ats<real_type>::log(abcS) +
                      Xp *
                        (Psri(0) * Psri(1) * t_1 * t_1 *
                           ats<real_type>::exp(-Psri(1) * t_1) -
                         ats<real_type>::exp(-t / Psri(2)) / Psri(2)) /
                        abcS);
            CrndDt = Ffac * PrDt + Prfac * FfacDt;
            Kokkos::parallel_for(Tines::RangeFactory<real_type>::TeamVectorRange(member, kmcd.nSpec),
                                 [&](const ordinal_type& k) {
                                   const real_type FfacDerk =
                                     Ffac * abcS * PrDer(k);
                                   CrndDer(k) =
                                     Ffac * PrDer(k) + Prfac * FfacDerk;
                                 });
          }
        } /* done with SRI form */
        /// TROE form
        else if (kmcd.reacPtype(ipfal) >= 3) {
          auto Ptroe = Kokkos::subview(rp, range_type(3, rp.extent(0)));
          const real_type ptroe_at_zero = Ptroe(0);
          const real_type one_minus_ptroe_at_zero = one - ptroe_at_zero;
          const bool ptroe_gt_zero = ptroe_at_zero > zero;
          const bool one_minus_ptroe_gt_zero = one_minus_ptroe_at_zero > zero;
          const real_type Fc1 =
            one_minus_ptroe_gt_zero
              ? one_minus_ptroe_at_zero * ats<real_type>::exp(-t / Ptroe(1))
              : zero;
          const real_type Fc2 =
            ptroe_gt_zero ? ptroe_at_zero * ats<real_type>::exp(-t / Ptroe(2))
                          : zero;

          real_type Fc(Fc1 + Fc2), FcDer(0);
          FcDer -= (one_minus_ptroe_gt_zero ? Fc1 / Ptroe(1) : zero);
          FcDer -= (ptroe_gt_zero ? Fc2 / Ptroe(2) : zero);

          if (kmcd.reacPtype(ipfal) == 4) {
            const real_type Fc3 = ats<real_type>::exp(-Ptroe(3) * t_1);
            Fc += Fc3;
            FcDer += Fc3 * Ptroe(3) * t_1 * t_1;
          }

          const real_type logFc =
            ats<real_type>::log(Fc) / ats<real_type>::log(real_type(10));
          const real_type logPr =
            ats<real_type>::log(Pr) / ats<real_type>::log(real_type(10));

          const bool Pr_gt_zero = Pr > zero;
          real_type Atroe(0), Btroe(0), Atroe_Btroe(0);
          if (Pr_gt_zero) {
            Atroe = logPr - 0.40 - 0.67 * logFc;
            Btroe = 0.75 - 1.27 * logFc - 0.14 * Atroe;
            Atroe_Btroe = Atroe / Btroe;
          } else {
            Atroe_Btroe = -one / 0.14;
          }

          const real_type oABtroe = one / (one + Atroe_Btroe * Atroe_Btroe);
          const real_type logFfac = logFc * oABtroe;
          real_type Ffac = ats<real_type>::pow(10, logFfac);

          real_type FfacDer0(0), FfacDer1(0);
          if (Pr_gt_zero) {
            const real_type oPr = one / (Pr * ats<real_type>::log(10));
            const real_type oFc = one / (Fc * ats<real_type>::log(10));
            const real_type Afc = -0.67 * oFc;
            const real_type Bfc = -1.1762 * oFc;
            const real_type Apr = 1.0 * oPr;
            const real_type Bpr = -0.14 * oPr;
            const real_type Gfac = Ffac * ats<real_type>::log(Fc) * 2.0 * Atroe /
                                   (Btroe * Btroe * Btroe) * oABtroe * oABtroe;

            /* dF/dPr */
            FfacDer0 = -Gfac * (Apr * Btroe - Bpr * Atroe);
            /* dF/dFc */
            FfacDer1 = Ffac / Fc * oABtroe - Gfac * (Afc * Btroe - Bfc * Atroe);
          }
          /* dF/dT */
          // const real_type dF_dt = FfacDer1 * FcDer + FfacDer0 * PrDt;
          const real_type Prfac = Pr / (one + Pr);
          //
          // CrndDt = Ffac * PrDt / ( (one + Pr) * (one + Pr) )  + Prfac * dF_dt;

          Ffac /= ((one + Pr) * (one + Pr));
          Ffac += Prfac * FfacDer0;

          CrndDt = Ffac * PrDt + Prfac * FfacDer1 * FcDer;

          Kokkos::parallel_for(
            Tines::RangeFactory<real_type>::TeamVectorRange(member, kmcd.nSpec),
            [&](const ordinal_type& k) {
              CrndDer(k) = Ffac * PrDer(k);
              // CrndDer(k) = PrDer(k) * ( Ffac / ((one + Pr) * (one + Pr)) +
              //                           Prfac * FfacDer0 );
                                       });
        } /* done with Troe form */
        ++ipfal;
      } /* done if pressure dependent reaction */
    }

#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("CrndDerivative.team_invoke.test.out", "a+");
      fprintf(fs, ":: CrndDerivative::team_invoke\n");

      fprintf(fs, ":::: input\n");
      fprintf(fs,
              "     t %e, nSpec %3d, nReac %3d, ireac %3d, itbdy %3d -> %3d, "
              "ipfal %3d -> %3d\n",
              t,
              kmcd.nSpec,
              kmcd.nReac,
              ireac,
              itbdy_prev,
              itbdy,
              ipfal_prev,
              ipfal);
      if (ireac == 0) {
        for (int i = 0; i < int(kfor.extent(0)); ++i)
          fprintf(fs, "     i %3d, kfor %e, concM %e\n", i, kfor(i), concM(i));
        for (int i = 0; i < int(concX.extent(0)); ++i)
          fprintf(fs, "     i %3d, concX %e\n", i, concX(i));
      }
      fprintf(fs, ":::: output\n");
      fprintf(fs,
              "     CrndDer extent %3d, CrndDt %e, PrDer extent %3d, PrDt %e\n",
              int(CrndDer.extent(0)),
              CrndDt,
              int(PrDer.extent(0)),
              PrDt);
      for (int i = 0; i < int(CrndDer.extent(0)); ++i)
        fprintf(
          fs, "     i %3d, CrndDer %e, PrDer %e\n", i, CrndDer(i), PrDer(i));
    }
#endif
  }
};

} // namespace Impl
} // namespace TChem

#endif
