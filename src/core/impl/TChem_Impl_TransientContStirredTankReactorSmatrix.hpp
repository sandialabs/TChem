/* =====================================================================================
TChem version 2.1.0
Copyright (2020) NTESS
https://github.com/sandialabs/TChem

Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains 
certain rights in this software.

This file is part of TChem. TChem is open-source software: you can redistribute it
and/or modify it under the terms of BSD 2-Clause License
(https://opensource.org/licenses/BSD-2-Clause). A copy of the license is also
provided under the main directory

Questions? Contact Cosmin Safta at <csafta@sandia.gov>, or
           Kyungjoo Kim at <kyukim@sandia.gov>, or
           Oscar Diaz-Ibarra at <odiazib@sandia.gov>

Sandia National Laboratories, Livermore, CA, USA
===================================================================================== */


#ifndef __TCHEM_IMPL_TCSTR_SMAT_HPP__
#define __TCHEM_IMPL_TCSTR_SMAT_HPP__

#include "TChem_Util.hpp"
namespace TChem {
namespace Impl {

struct TransientContStirredTankReactorSmatrix
{

  template<typename MemberType,
           typename RealType1DViewType,
           typename RealType2DViewType,
           typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType,
           typename ContStirredTankReactorConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    /// input
    const real_type& t,
    const RealType1DViewType& Ys, /// (kmcd.nSpec) mass fraction
    const RealType1DViewType& Zs, // (kmcdSurf.nSpec) site fraction
    const real_type& density,
    const real_type& p,   // pressure
    /// output
    const RealType2DViewType& Smat,
    const RealType2DViewType& Ssmat,
    const RealType1DViewType& Sconv,
    // work
    const RealType1DViewType& hks,
    const RealType1DViewType& cpks,
    const RealType1DViewType& sumSpecSurf,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf,
    // const input for plug flow reactor
    const ContStirredTankReactorConstDataType& cstr)
  {

    // printf("mdotIn %e Vol %e Acat %e\n", cstr.mdotIn, cstr.Vol, cstr.Acat);

    const real_type Nvars = kmcd.nSpec + 1; // T and mass fraction
    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, kmcd.nReac),
                         [&](const ordinal_type& i) {
                           for (ordinal_type j = 0; j < Nvars; ++j) {
                             Smat(j, i) = 0;
                           }
                         }); /* done loop over all reactions */

    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, kmcdSurf.nReac),
                         [&](const ordinal_type& i) {
                           for (ordinal_type j = 0; j < Nvars; ++j) {
                             Ssmat(j, i) = 0;
                           }
                           sumSpecSurf(i) = 0;
                         }); /* done loop over all reactions */

    /// 1. compute species enthalies
    EnthalpySpecMs ::team_invoke(member, t, hks, cpks, kmcd);

    /// 2. cpmix
    const real_type cpmix = CpMixMs::team_invoke(member, t, Ys, cpks, kmcd);

    const real_type ConEnergy = -1. / (cpmix * density );
    const real_type ConSpecies = 1. / density ;

    member.team_barrier();


    real_type enthapyMix(0);
    Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& k, real_type& update) {
        update += hks(k) * Ys(k);

    },
    enthapyMix);

    real_type enthapyMix2(0);
    Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& k, real_type& update) {
        update += hks(k) * cstr.Yi(k);

    },
    enthapyMix2);
    /* assemble matrix based on integer stoichiometric coefficients */

    for (ordinal_type i = 0; i < kmcd.nReac; i++) {
      // energy
      // reactans
      real_type sumEnerR(0);
      Kokkos::parallel_reduce(
        Kokkos::TeamVectorRange(member, kmcd.reacNreac(i)),
        [&](const ordinal_type& j, real_type& update) {
          const ordinal_type kspec = kmcd.reacSidx(i, j);
          update += hks(kspec) * kmcd.sMass(kspec) * kmcd.reacNuki(i, j);
        },
        sumEnerR);

      // products
      const ordinal_type joff = kmcd.reacSidx.extent(1) / 2;
      real_type sumEnerP(0);
      Kokkos::parallel_reduce(
        Kokkos::TeamVectorRange(member, kmcd.reacNprod(i)),
        [=](const ordinal_type& j, real_type& update) {
          const ordinal_type kspec = kmcd.reacSidx(i, j + joff);
          update += hks(kspec) * kmcd.sMass(kspec) * kmcd.reacNuki(i, j + joff);
        },
        sumEnerP);

      Smat(0, i) = (sumEnerP + sumEnerR) * ConEnergy;


      // species equations

      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, kmcd.reacNreac(i)),
                           [&](const ordinal_type& j) {
                             const ordinal_type kspec = kmcd.reacSidx(i, j);
                             Smat(kspec + 1, i) = kmcd.sMass(kspec) *
                                                  kmcd.reacNuki(i, j) *
                                                  ConSpecies;
                             // Kokkos::atomic_fetch_add(&Smat(kspec+ 1,i),
                             // val);
                           });

      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, kmcd.reacNprod(i)),
        [=](const ordinal_type& j) {
          const ordinal_type kspec = kmcd.reacSidx(i, j + joff);
          Smat(kspec + 1, i) =
            kmcd.sMass(kspec) * kmcd.reacNuki(i, j + joff) * ConSpecies;
          // Kokkos::atomic_fetch_add(&Smat(kspec+ 1,i), val);
        });

    } /* done loop over all reactions */

    member.team_barrier();

    const real_type ConEnergySurf =
      -cstr.Acat * enthapyMix / (density  * cstr.Vol * cpmix);

    const real_type ConSpeciesSurf = cstr.Acat  / (density  * cstr.Vol);

    //terms that do not depend directly on surface or gas reaction mechanism
    //energy
    // Sconv(0) = cstr.mdotIn * (cstr.EnthalpyIn -  enthapyMix )/ (cpmix * density * cstr.Vol );
    Sconv(0) =  cstr.mdotIn * (cstr.EnthalpyIn - enthapyMix2 )/(density  * cstr.Vol * cpmix);
    //species
    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, kmcd.nSpec),
                         [&](const ordinal_type& k) {

      Sconv(k+1) = cstr.mdotIn * (cstr.Yi(k) -  Ys(k))/ (density  * cstr.Vol);

    });


    for (ordinal_type i = 0; i < kmcdSurf.nReac; i++) {

      real_type sumWkVkiR(0);
      Kokkos::parallel_reduce(
        Kokkos::TeamVectorRange(member, kmcdSurf.reacNreac(i)),
        [&](const ordinal_type& j, real_type& update) {
          if (kmcdSurf.reacSsrf(i, j) == 0) { // only gas species
            const ordinal_type kspec = kmcdSurf.reacSidx(i, j);
            update += kmcd.sMass(kspec) * kmcdSurf.reacNuki(i, j);
          }
        },
        sumWkVkiR);

      // products
      // sum_{k=1}^nSpec {W_k * V_{ki}}
      const ordinal_type joff = kmcdSurf.reacSidx.extent(1) / 2;
      real_type sumWkVkiP(0);
      Kokkos::parallel_reduce(
        Kokkos::TeamVectorRange(member, kmcdSurf.reacNprod(i)),
        [=](const ordinal_type& j, real_type& update) {
          if (kmcdSurf.reacSsrf(i, j + joff) == 0) { // only gas species
            const ordinal_type kspec = kmcdSurf.reacSidx(i, j + joff);
            update += kmcd.sMass(kspec) * kmcdSurf.reacNuki(i, j + joff);
          }
        },
        sumWkVkiP);



      // species equations
      // W \dot V_ki
      sumSpecSurf(i) = (sumWkVkiR + sumWkVkiP) * ConSpeciesSurf;

      real_type sumWkVkiRh(0);
      Kokkos::parallel_reduce(
        Kokkos::TeamVectorRange(member, kmcdSurf.reacNreac(i)),
        [&](const ordinal_type& j, real_type& update) {
          if (kmcdSurf.reacSsrf(i, j) == 0) { // only gas species
            const ordinal_type kspec = kmcdSurf.reacSidx(i, j);
            update += hks(kspec)*kmcd.sMass(kspec) * kmcdSurf.reacNuki(i, j);
          }
        },
        sumWkVkiRh);

      // products
      // sum_{k=1}^nSpec {W_k * V_{ki}}
      // const ordinal_type joff = kmcdSurf.reacSidx.extent(1) / 2;
      real_type sumWkVkiPh(0);
      Kokkos::parallel_reduce(
        Kokkos::TeamVectorRange(member, kmcdSurf.reacNprod(i)),
        [=](const ordinal_type& j, real_type& update) {
          if (kmcdSurf.reacSsrf(i, j + joff) == 0) { // only gas species
            const ordinal_type kspec = kmcdSurf.reacSidx(i, j + joff);
            update += hks(kspec)*kmcd.sMass(kspec) * kmcdSurf.reacNuki(i, j + joff);
          }
        },
        sumWkVkiPh);

        // energy
        //hA\sum_sk W/cp rho V => AH/cp/rho V Wvki
        Ssmat(0, i) = (sumWkVkiR + sumWkVkiP) * ConEnergySurf
        + (sumWkVkiPh+sumWkVkiRh)* ConEnergySurf/enthapyMix;


    } /* done loop over all reactions */

    // species
    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(member, kmcd.nSpec), [&](const ordinal_type& k) {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, kmcdSurf.nReac),
                             [&](const ordinal_type& i) {
                               Ssmat(k + 1, i) = -Ys(k) * sumSpecSurf(i);
                             }); /* done loop over all reactions */
      });                        /* done loop over all species */

    member.team_barrier();

    Kokkos::parallel_for(
      Kokkos::TeamVectorRange(member, kmcdSurf.nReac),
      [&](const ordinal_type& i) {
        // reactans
        for (ordinal_type j = 0; j < kmcdSurf.reacNreac(i); ++j) {
          if (kmcdSurf.reacSsrf(i, j) == 0) { // only gas species
            const ordinal_type kspec = kmcdSurf.reacSidx(i, j);
            const real_type val =
              kmcd.sMass(kspec) * kmcdSurf.reacNuki(i, j) * ConSpeciesSurf;
            // Ssmat(kspec + 1,i) += val;
            Kokkos::atomic_fetch_add(&Ssmat(kspec + 1, i), val);
          }
        }

        // check if this an issue  race condition
        const ordinal_type joff = kmcdSurf.maxSpecInReac / 2;
        for (ordinal_type j = 0; j < kmcdSurf.reacNprod(i); ++j) {
          if (kmcdSurf.reacSsrf(i, j + joff) == 0) { // only gas species
            const ordinal_type kspec = kmcdSurf.reacSidx(i, j + joff);
            const real_type val = kmcd.sMass(kspec) *
                                  kmcdSurf.reacNuki(i, j + joff) *
                                  ConSpeciesSurf;
            Kokkos::atomic_fetch_add(&Ssmat(kspec + 1, i), val);
            // Ssmat(kspec + 1,i) += val;
          }
        }
      });

    member.team_barrier();

    //energy
    for (ordinal_type i = 0; i < kmcdSurf.nReac; i++) {

      real_type sumhkYkvik(0);
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(member, kmcd.nSpec),
                           [&](const ordinal_type& k, real_type& update) {
                             update += hks(k) * Ys(k) * sumSpecSurf(i);

                           },sumhkYkvik);
    //sum_{k=1}^{kg} (hk*Yk*A*sk*wk) // 
    real_type val = sumhkYkvik/cpmix;
    Kokkos::atomic_fetch_add(&Ssmat(0, i), val);

    }


    member.team_barrier();



#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("TransientContStirredTankReactorSmatrix.team_invoke.test.out", "a+");
      fprintf(fs, ":: TransientContStirredTankReactorSmatrix::team_invoke\n");
      fprintf(fs, ":::: input\n");
      fprintf(fs,
              "     nSpec %3d, nReac %3d, site density %e\n",
              kmcdSurf.nSpec,
              kmcdSurf.nReac,
              kmcdSurf.sitedensity);
      fprintf(fs, "  t %e, p %e\n", t, p);
      for (int i = 0; i < kmcd.nSpec; ++i)
        fprintf(fs, "   i %3d,  Ys %e, \n", i, Ys(i));
      for (int i = 0; i < kmcdSurf.nSpec; ++i)
        fprintf(fs, "   i %3d,  Zs %e, \n", i, Zs(i));

      fprintf(fs, "stoichiometric matrix gas species in surface phase \n");
      for (int i = 0; i < kmcd.nSpec; ++i) {
        for (int j = 0; j < kmcdSurf.nReac; j++) {
          fprintf(fs, "%d ", kmcdSurf.vki(i, j));
        }
        fprintf(fs, "\n");
      }

      fprintf(fs, ":::: output\n");
      for (int i = 0; i < Nvars; ++i)
        for (int j = 0; j < kmcd.nReac; j++) {
          fprintf(fs, "   i %d, j %d,  Smat %e, \n", i, j, Smat(i, j));
        }
      for (int i = 0; i < Nvars; ++i)
        for (int j = 0; j < kmcdSurf.nReac; j++) {
          fprintf(fs, "   i %d, j %d, Ssmat %e, \n", i, j, Ssmat(i, j));
        }
    }
#endif
  }

  template<typename MemberType,
           typename WorkViewType,
           typename RealType1DViewType,
           typename RealType2DViewType,
           typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType,
           typename ContStirredTankReactorConstDataType>
  KOKKOS_FORCEINLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const real_type& t,
    const RealType1DViewType& Ys, /// (kmcd.nSpec)
    const RealType1DViewType& Zs, // (kmcdSurf.nSpec) site fraction
    const real_type& density,
    const real_type& p, // pressure
    /// output
    const RealType2DViewType& Smat,//gas terms
    const RealType2DViewType& Ssmat,//surface terms
    const RealType1DViewType& Sconv,//other terms
    // work
    /// workspace
    const WorkViewType& work,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf,
    const ContStirredTankReactorConstDataType& cstr)
  {

    auto w = (real_type*)work.data();
    auto hks = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto cpks = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto sumSpecSurf = RealType1DViewType(w, kmcdSurf.nReac);
    w += kmcdSurf.nSpec;

    team_invoke_detail(member,
                       t,
                       Ys,
                       Zs,
                       density,
                       p,
                       Smat,
                       Ssmat,
                       Sconv,
                       hks,
                       cpks,
                       sumSpecSurf,
                       kmcd,
                       kmcdSurf,
                       cstr);
  }
};

} // namespace Impl
} // namespace TChem

#endif
