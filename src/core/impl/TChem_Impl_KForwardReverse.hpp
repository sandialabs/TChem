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
#ifndef __TCHEM_IMPL_KFORWARDREVERSE_HPP__
#define __TCHEM_IMPL_KFORWARDREVERSE_HPP__

#include "TChem_Impl_SumNuGk.hpp"
#include "TChem_Impl_SumRealNuGk.hpp"
#include "TChem_Util.hpp"
// #define TCHEM_ENABLE_SERIAL_TEST_OUTPUT
namespace TChem {
namespace Impl {

struct KForwardReverse
{
  template<typename MemberType,
           typename RealType1DViewType,
           typename OrdinalType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    /// input temperature
    const real_type& t,
    const real_type& p,
    const RealType1DViewType& gk,
    /// output
    const RealType1DViewType& kfor,
    const RealType1DViewType& krev,
    /// workspace
    const OrdinalType1DViewType& iplogs,
    const OrdinalType1DViewType& irevs,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    Kokkos::single(Kokkos::PerTeam(member), [&]() {
      /// compute iterators
      ordinal_type iplog(0), irev(0);
      for (ordinal_type i = 0; i < kmcd.nReac; ++i) {
        iplogs(i) = iplog;
        iplog += (iplog < kmcd.nPlogReac) && (i == kmcd.reacPlogIdx(iplog));

        irevs(i) = irev;
        irev += (kmcd.isRev(i)) && (irev < kmcd.nRevReac) &&
                (kmcd.reacRev(irev) == i);
      }
    });
    member.team_barrier();

    const real_type zero(0);
    const real_type t_1 = real_type(1) / t;
    const real_type tln = ats<real_type>::log(t);
    const real_type logP =
      kmcd.nPlogReac > 0 ? ats<real_type>::log(p / ATMPA) : real_type(0);

    ///
    /// this loop has an sparse access structure with an incremental indices
    /// which is not feasible to parallelize over a team
    ///
    Kokkos::parallel_for(
      Kokkos::TeamVectorRange(member, kmcd.nReac), [&](const ordinal_type& i) {
        const ordinal_type iplog = iplogs(i);
        const ordinal_type irev = irevs(i);
        const bool plogtest =
          (iplog < kmcd.nPlogReac) && (i == kmcd.reacPlogIdx(iplog));

        ///
        /// plog test
        ///
        if (plogtest) {
          ordinal_type idx = kmcd.reacPlogPno(iplog);
          // printf("idx %d\n", idx );
          auto rpp = Kokkos::subview(kmcd.reacPlogPars, 0, Kokkos::ALL());
          // printf("reacPlogPars %e log %e\n",kmcd.reacPlogPars(idx,0), logP );
          if (logP <= kmcd.reacPlogPars(idx, 0)) {
            rpp.assign_data(&kmcd.reacPlogPars(idx, 0));
            kfor(i) = ats<real_type>::exp(rpp(1) + rpp(2) * tln - rpp(3) * t_1);
            // printf("ki i %d,  ki1 %e, logPi %e, log(A) %e, b %e, Ea %e \n",i,
            // kfor(i), rpp(0),rpp(1),rpp(2),rpp(3) );

          } else {
            idx = kmcd.reacPlogPno(iplog + 1) - 1;
            // printf("reacPlogPars %e log %e\n",kmcd.reacPlogPars(idx,0), logP
            // );
            rpp.assign_data(&kmcd.reacPlogPars(idx, 0));
            if (logP >= kmcd.reacPlogPars(idx, 0)) {
              kfor(i) =
                ats<real_type>::exp(rpp(1) + rpp(2) * tln - rpp(3) * t_1);
              // printf("ki i %d,  ki1 %e, logPi %e, log(A) %e, b %e, Ea %e
              // \n",i, kfor(i), rpp(0),rpp(1),rpp(2),rpp(3) );
            } else {
              // printf("Reac No %d, kmcd.reacPlogPno(iplog) %d,
              // kmcd.reacPlogPno(iplog+1)-1 %d \n", i, kmcd.reacPlogPno(iplog),
              // kmcd.reacPlogPno(iplog+1)-1 );
              for (ordinal_type j = kmcd.reacPlogPno(iplog);
                   j < kmcd.reacPlogPno(iplog + 1);
                   ++j) {
                // printf("logP %e kmcd.reacPlogPars(j,0)%e \n", logP,
                // kmcd.reacPlogPars(j,0)  );
                if (logP <= kmcd.reacPlogPars(j, 0)) {
                  // printf("between intervals logP %e kmcd.reacPlogPars(j,0) %e
                  // \n",logP,  kmcd.reacPlogPars(j,0) );
                  rpp.assign_data(&kmcd.reacPlogPars(j, 0));
                  const real_type ki1 = (rpp(1) + rpp(2) * tln - rpp(3) * t_1);
                  const real_type rpp1 = rpp(0);
                  // printf("ki1 i %d,  ki1 %e, logPi %e, log(A) %e, b %e, Ea %e
                  // \n",i, ki1, rpp(0),rpp(1),rpp(2),rpp(3) );
                  rpp.assign_data(&kmcd.reacPlogPars(j - 1, 0));
                  const real_type ki = (rpp(1) + rpp(2) * tln - rpp(3) * t_1);
                  // printf("ki i %d,  ki1 %e, logPi %e, log(A) %e, b %e, Ea %e
                  // \n",i, ki, rpp(0),rpp(1),rpp(2),rpp(3) );

                  kfor(i) = ats<real_type>::exp(
                    ki + (logP - rpp(0)) * (ki1 - ki) / (rpp1 - rpp(0)));
                  // printf("Reacton No %d  kfor PLOG %e\n",i, kfor(i) );
                  break;
                }
              } /* Done loop over all intervals */
            }   /* Done with branch pmin<p<pmax */
          }     /* Done with branch p>pmin */
        }       /* Done if reaction has a PLOG form */
        else {
          kfor(i) = (kmcd.reacArhenFor(i, 0) *
                     ats<real_type>::exp(kmcd.reacArhenFor(i, 1) * tln -
                                         kmcd.reacArhenFor(i, 2) * t_1));
        }

        ///
        /// check reverse reaction
        ///

        krev(i) = zero;
        /* is reaction reversible ? */
        if (kmcd.isRev(i)) {
          /* are reverse Arhenius parameters given ? */
          const bool is_arhenius_parameters_given =
            (irev < kmcd.nRevReac && kmcd.reacRev(irev) == i);

          if (is_arhenius_parameters_given) {
            /* yes, reverse Arhenius parameters are given */
            krev(i) =
              (kmcd.reacArhenRev(irev, 0) < ats<real_type>::epsilon()
                 ? zero
                 : kmcd.reacArhenRev(irev, 0) *
                     ats<real_type>::exp(kmcd.reacArhenRev(irev, 1) * tln -
                                         kmcd.reacArhenRev(irev, 2) * t_1));
          } /* done if section for reverse Arhenius parameters */
          else {
            /* no, need to compute equilibrium constant */
            const ordinal_type ir = kmcd.reacScoef(i);
            const real_type sumNuGk =
              ir == -1 ? SumNuGk::serial_invoke(i, gk, kmcd)
                       : SumRealNuGk::serial_invoke(i, ir, gk, kmcd);
            const real_type kc =
              kmcd.kc_coeff(i) * ats<real_type>::exp(sumNuGk);
            krev(i) = kfor(i) / kc;
          } /* done if section for equilibrium constant */
        }   /* done if reaction is reversible */
      });   /* done computing kforward and kreverse rate constants */
#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("KForwardReverse.team_invoke.test.out", "a+");
      fprintf(fs, ":: KForwardReverse::team_invoke\n");
      fprintf(fs, ":::: input\n");
      fprintf(fs,
              "     nSpec %3d, nReac %3d, t %e, p %e\n",
              kmcd.nSpec,
              kmcd.nReac,
              t,
              p);
      fprintf(fs, ":::: output\n");
      for (int i = 0; i < int(kfor.extent(0)); ++i)
        fprintf(fs, "     i %3d, kfor %e, krev %e\n", i, kfor(i), krev(i));
    }
#endif
  }

  template<typename MemberType,
           typename WorkViewType,
           typename RealType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_FORCEINLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input temperature
    const real_type& t,
    const real_type& p,
    const RealType1DViewType& gk,
    /// output
    const RealType1DViewType& kfor,
    const RealType1DViewType& krev,
    /// workspace
    const WorkViewType& work,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    auto w = (ordinal_type*)work.data();
    auto iplogs =
      Kokkos::View<ordinal_type*,
                   Kokkos::LayoutRight,
                   typename WorkViewType::memory_space>(w, kmcd.nReac);
    w += kmcd.nReac;
    auto irevs =
      Kokkos::View<ordinal_type*,
                   Kokkos::LayoutRight,
                   typename WorkViewType::memory_space>(w, kmcd.nReac);
    w += kmcd.nReac;
    team_invoke_detail(member, t, p, gk, kfor, krev, iplogs, irevs, kmcd);
  }
};

struct KForwardReverseDerivative
{
  template<typename MemberType,
           typename WorkViewType,
           typename RealType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input temperature
    const real_type& t,
    const real_type& p,
    /// output
    const RealType1DViewType& kforp,
    const RealType1DViewType& krevp,
    /// workspace
    const RealType1DViewType& gkp,
    const WorkViewType& work,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    auto w = (ordinal_type*)work.data();
    auto iplogs =
      Kokkos::View<ordinal_type*,
                   Kokkos::LayoutRight,
                   typename WorkViewType::memory_space>(w, kmcd.nReac);
    w += kmcd.nReac;
    auto irevs =
      Kokkos::View<ordinal_type*,
                   Kokkos::LayoutRight,
                   typename WorkViewType::memory_space>(w, kmcd.nReac);
    w += kmcd.nReac;
    Kokkos::single(Kokkos::PerTeam(member), [&]() {
      /// compute iterators
      ordinal_type iplog(0), irev(0);
      for (ordinal_type i = 0; i < kmcd.nReac; ++i) {
        iplogs(i) = iplog;
        iplog += (iplog < kmcd.nPlogReac) && (i == kmcd.reacPlogIdx(iplog));

        irevs(i) = irev;
        irev += (kmcd.isRev(i)) && (irev < kmcd.nRevReac) &&
                (kmcd.reacRev(irev) == i);
      }
    });
    member.team_barrier();

    const real_type zero(0);
    const real_type t_1 = real_type(1) / t;
    const real_type logP =
      kmcd.nPlogReac > 0 ? ats<real_type>::log(p / ATMPA) : real_type(0);

    ///
    /// this loop has an sparse access structure with an incremental indices
    /// which is not feasible to parallelize over a team
    ///
    Kokkos::parallel_for(
      Kokkos::TeamVectorRange(member, kmcd.nReac), [&](const ordinal_type& i) {
        const ordinal_type iplog = iplogs(i);
        const ordinal_type irev = irevs(i);
        const bool plogtest =
          (iplog < kmcd.nPlogReac) && (i == kmcd.reacPlogIdx(iplog));

        ///
        /// plog test
        ///
        if (plogtest) {
          ordinal_type idx = kmcd.reacPlogPno(iplog);
          auto rpp = Kokkos::subview(kmcd.reacPlogPars, 0, Kokkos::ALL());
          if (logP <= kmcd.reacPlogPars(idx, 0)) {
            rpp.assign_data(&kmcd.reacPlogPars(idx, 0));
            kforp(i) = t_1 * (rpp(2) + rpp(3) * t_1);
          } else {
            idx = kmcd.reacPlogPno(iplog + 1) - 1;
            rpp.assign_data(&kmcd.reacPlogPars(idx, 0));
            if (logP >= kmcd.reacPlogPars(idx, 0)) {
              kforp(i) = t_1 * (rpp(2) + rpp(3) * t_1);
            } else {
              for (ordinal_type j = kmcd.reacPlogPno(iplog);
                   j < kmcd.reacPlogPno(iplog + 1);
                   ++j) {
                if (logP <= kmcd.reacPlogPars(j, 0)) {
                  rpp.assign_data(&kmcd.reacPlogPars(j, 0));
                  const real_type ki1 = t_1 * (rpp(2) + rpp(3) * t_1);
                  const real_type rpp1 = rpp(0);

                  rpp.assign_data(&kmcd.reacPlogPars(j - 1, 0));
                  const real_type ki = t_1 * (rpp(2) + rpp(3) * t_1);

                  kforp(i) =
                    ki + (logP - rpp(0)) / (rpp1 - rpp(0)) * (ki1 - ki);
                  break;
                }
              } /* Done loop over all intervals */
            }   /* Done with branch pmin<p<pmax */
          }     /* Done with branch p>pmin */
        }       /* Done if reaction has a PLOG form */
        else {
          kforp(i) =
            (t_1 * (kmcd.reacArhenFor(i, 1) + kmcd.reacArhenFor(i, 2) * t_1));
        }

        ///
        /// check reverse reaction
        ///
        krevp(i) = zero;
        /* is reaction reversible ? */
        if (kmcd.isRev(i)) {
          /* are reverse Arhenius parameters given ? */
          const bool is_arhenius_parameters_given =
            (irev < kmcd.nRevReac && kmcd.reacRev(irev) == i);

          if (is_arhenius_parameters_given) {
            /* yes, reverse Arhenius parameters are given */
            krevp(i) = t_1 * (kmcd.reacArhenRev(irev, 1) +
                              kmcd.reacArhenRev(irev, 2) * t_1);
          } /* done if section for reverse Arhenius parameters */
          else {
            /* no, need to compute equilibrium constant */
            const ordinal_type ir = kmcd.reacScoef(i);
            const real_type sumNuGkp =
              ir == -1 ? SumNuGk::serial_invoke(i, gkp, kmcd)
                       : SumRealNuGk::serial_invoke(i, ir, gkp, kmcd);
            krevp(i) = kforp(i) - sumNuGkp;
          } /* done if section for equilibrium constant */
        }
      }); /* done computing kforward and kreverse rate constants */
#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("KForwardReverseDerivative.team_invoke.test.out", "a+");
      fprintf(fs, ":: KForwardReverseDerivative::team_invoke\n");
      fprintf(fs, ":::: input\n");
      fprintf(fs,
              "     nSpec %3d, nReac %3d, t %e, p %e\n",
              kmcd.nSpec,
              kmcd.nReac,
              t,
              p);
      fprintf(fs, ":::: output\n");
      for (int i = 0; i < int(kforp.extent(0)); ++i)
        fprintf(fs, "     i %3d, kforp %e, krevp %e\n", i, kforp(i), krevp(i));
    }
#endif
  }
};

} // namespace Impl
} // namespace TChem

#endif
