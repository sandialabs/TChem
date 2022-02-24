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
#ifndef __TCHEM_IMPL_TCSTR_SMAT_HPP__
#define __TCHEM_IMPL_TCSTR_SMAT_HPP__

#include "TChem_Util.hpp"
namespace TChem {
namespace Impl {

template<typename ValueType, typename DeviceType>
struct TransientContStirredTankReactorSmatrix
{

  using value_type = ValueType;
  using device_type = DeviceType;
  using scalar_type = typename ats<value_type>::scalar_type;

  using real_type = scalar_type;
  using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;
  using real_type_2d_view_type = Tines::value_type_2d_view<real_type,device_type>;

  using kinetic_model_type      = KineticModelConstData<device_type>;
  using kinetic_surf_model_type = KineticSurfModelConstData<device_type>;
  using cstr_data_type = TransientContStirredTankReactorData<device_type>;


  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    /// input
    const real_type& t,
    const real_type_1d_view_type& Ys, /// (kmcd.nSpec) mass fraction
    const real_type_1d_view_type& Zs, // (kmcdSurf.nSpec) site fraction
    const real_type& density,
    const real_type& p,   // pressure
    /// output
    const real_type_2d_view_type& Smat,
    const real_type_2d_view_type& Ssmat,
    const real_type_1d_view_type& Sconv,
    // work
    const real_type_1d_view_type& hks,
    const real_type_1d_view_type& cpks,
    const real_type_1d_view_type& sumSpecSurf,
    /// const input from kinetic model
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf,
    // const input for cstr
    const cstr_data_type& cstr)
  {

    using EnthalpySpecMs = EnthalpySpecMsFcn<real_type,device_type>;
    using CpMixMs = CpMixMs<real_type, device_type>;

    // printf("mdotIn %e Vol %e Acat %e\n", cstr.mdotIn, cstr.Vol, cstr.Acat);

    const real_type Nvars = kmcd.nSpec + 1; // T and mass fraction
    Kokkos::parallel_for(Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nReac),
                         [&](const ordinal_type& i) {
                           for (ordinal_type j = 0; j < Nvars; ++j) {
                             Smat(j, i) = 0;
                           }
                         }); /* done loop over all reactions */

    Kokkos::parallel_for(Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcdSurf.nReac),
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
      Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& k, real_type& update) {
        update += hks(k) * Ys(k);

    },
    enthapyMix);

    real_type enthapyMix2(0);
    Kokkos::parallel_reduce(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
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
        Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.reacNreac(i)),
        [&](const ordinal_type& j, real_type& update) {
          const ordinal_type kspec = kmcd.reacSidx(i, j);
          update += hks(kspec) * kmcd.sMass(kspec) * kmcd.reacNuki(i, j);
        },
        sumEnerR);

      // products
      const ordinal_type joff = kmcd.reacSidx.extent(1) / 2;
      real_type sumEnerP(0);
      Kokkos::parallel_reduce(
        Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.reacNprod(i)),
        [=](const ordinal_type& j, real_type& update) {
          const ordinal_type kspec = kmcd.reacSidx(i, j + joff);
          update += hks(kspec) * kmcd.sMass(kspec) * kmcd.reacNuki(i, j + joff);
        },
        sumEnerP);

      Smat(0, i) = (sumEnerP + sumEnerR) * ConEnergy * cstr.isothermal ;


      // species equations

      Kokkos::parallel_for(Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.reacNreac(i)),
                           [&](const ordinal_type& j) {
                             const ordinal_type kspec = kmcd.reacSidx(i, j);
                             Smat(kspec + 1, i) = kmcd.sMass(kspec) *
                                                  kmcd.reacNuki(i, j) *
                                                  ConSpecies;
                           });

      Kokkos::parallel_for(
        Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.reacNprod(i)),
        [=](const ordinal_type& j) {
          const ordinal_type kspec = kmcd.reacSidx(i, j + joff);
          const real_type val =
            kmcd.sMass(kspec) * kmcd.reacNuki(i, j + joff) * ConSpecies;
          Kokkos::atomic_fetch_add(&Smat(kspec + 1, i), val);
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
    Kokkos::parallel_for(Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
                         [&](const ordinal_type& k) {

      Sconv(k+1) = cstr.mdotIn * (cstr.Yi(k) -  Ys(k))/ (density  * cstr.Vol);

    });


    for (ordinal_type i = 0; i < kmcdSurf.nReac; i++) {

      real_type sumWkVkiR(0);
      Kokkos::parallel_reduce(
        Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcdSurf.reacNreac(i)),
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
        Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcdSurf.reacNprod(i)),
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
        Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcdSurf.reacNreac(i)),
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
        Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcdSurf.reacNprod(i)),
        [=](const ordinal_type& j, real_type& update) {
          if (kmcdSurf.reacSsrf(i, j + joff) == 0) { // only gas species
            const ordinal_type kspec = kmcdSurf.reacSidx(i, j + joff);
            update += hks(kspec)*kmcd.sMass(kspec) * kmcdSurf.reacNuki(i, j + joff);
          }
        },
        sumWkVkiPh);

        // energy
        //hA\sum_sk W/cp rho V => AH/cp/rho V Wvki
        Ssmat(0, i) = ( (sumWkVkiR + sumWkVkiP) * ConEnergySurf
        + (sumWkVkiPh+sumWkVkiRh)* ConEnergySurf/enthapyMix ) * cstr.isothermal ;


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
      Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcdSurf.nReac),
      [&](const ordinal_type& i) {
        // reactans
        for (ordinal_type j = 0; j < kmcdSurf.reacNreac(i); ++j) {
          if (kmcdSurf.reacSsrf(i, j) == 0) { // only gas species
            const ordinal_type kspec = kmcdSurf.reacSidx(i, j);
            const real_type val =
              kmcd.sMass(kspec) * kmcdSurf.reacNuki(i, j) * ConSpeciesSurf;
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
    real_type val = cstr.isothermal * sumhkYkvik/cpmix;
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
           typename WorkViewType>
  KOKKOS_FORCEINLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const real_type& t,
    const real_type_1d_view_type& Ys, /// (kmcd.nSpec)
    const real_type_1d_view_type& Zs, // (kmcdSurf.nSpec) site fraction
    const real_type& density,
    const real_type& p, // pressure
    /// output
    const real_type_2d_view_type& Smat,//gas terms
    const real_type_2d_view_type& Ssmat,//surface terms
    const real_type_1d_view_type& Sconv,//other terms
    // work
    /// workspace
    const WorkViewType& work,
    /// const input from kinetic model
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf,
    // const input for cstr
    const cstr_data_type& cstr)
  {

    auto w = (real_type*)work.data();
    auto hks = real_type_1d_view_type(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto cpks = real_type_1d_view_type(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto sumSpecSurf = real_type_1d_view_type(w, kmcdSurf.nReac);
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
