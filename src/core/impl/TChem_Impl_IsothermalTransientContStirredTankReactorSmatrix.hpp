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
#ifndef __TCHEM_IMPL_ISOTHERMAL_TCSTR_SMAT_HPP__
#define __TCHEM_IMPL_ISOTHERMAL_TCSTR_SMAT_HPP__

#include "TChem_Util.hpp"
namespace TChem {
namespace Impl {

template<typename ValueType, typename DeviceType>
struct IsothermalTransientContStirredTankReactorSmatrix
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
    const real_type_2d_view_type& Sg,
    const real_type_2d_view_type& Ss,
    const real_type_2d_view_type& Sconv,
    const real_type_2d_view_type& Cmat,
    // work
    const real_type_1d_view_type& sumSpecSurf,
    /// const input from kinetic model
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf,
    // const input for cstr
    const cstr_data_type& cstr)
  {

    const ordinal_type idx = cstr.poisoning_species_idx;
    const real_type Acat = idx > 0 ?
                            cstr.Acat * (real_type(1) - Zs(idx)) : cstr.Acat;

    // printf("mdotIn %e Vol %e Acat %e\n", cstr.mdotIn, cstr.Vol, Acat);
    // // do I need this?
    // Kokkos::parallel_for(Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nReac),
    //                      [&](const ordinal_type& i) {
    //                        for (ordinal_type j = 0; j < kmcd.nSpec ; ++j) {
    //                          Sg(j, i) = 0;
    //                        }
    //                      }); /* done loop over all reactions */
    //
    // Kokkos::parallel_for(Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcdSurf.nReac),
    //                      [&](const ordinal_type& i) {
    //                        for (ordinal_type j = 0; j < kmcd.nSpec ; ++j) {
    //                          Ss(j, i) = 0;
    //                        }
    //                        sumSpecSurf(i) = 0;
    //                      }); /* done loop over all reactions */
    //
    const real_type ConSpecies = 1. / density ;
    //
    member.team_barrier();

    /* assemble matrix based on integer stoichiometric coefficients */

    for (ordinal_type i = 0; i < kmcd.nReac; i++) {

      // species equations
      // reactans
      Kokkos::parallel_for(Tines::RangeFactory<real_type>::TeamVectorRange(member, kmcd.reacNreac(i)),
                           [&](const ordinal_type& j) {
                             const ordinal_type kspec = kmcd.reacSidx(i, j);
                             Sg(kspec, i) = kmcd.sMass(kspec) *
                                                  kmcd.reacNuki(i, j) *
                                                  ConSpecies;
                           });
      // products
      const ordinal_type joff = kmcd.reacSidx.extent(1) / 2;
      Kokkos::parallel_for(
        Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.reacNprod(i)),
        [=](const ordinal_type& j) {
          const ordinal_type kspec = kmcd.reacSidx(i, j + joff);
          const real_type val =
            kmcd.sMass(kspec) * kmcd.reacNuki(i, j + joff) * ConSpecies;
          Kokkos::atomic_fetch_add(&Sg(kspec, i), val);
        });

    } /* done loop over all reactions */



    const real_type ConSpeciesSurf = Acat  / (density  * cstr.Vol);
    //species
    Kokkos::parallel_for(Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
                         [&](const ordinal_type& k) {

      Sconv(k,0) = cstr.mdotIn * (cstr.Yi(k) -  Ys(k))/ (density  * cstr.Vol);

    });

    member.team_barrier();


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

    } /* done loop over all reactions */
    member.team_barrier();

    // species
    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(member, kmcd.nSpec), [&](const ordinal_type& k) {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, kmcdSurf.nReac),
                             [&](const ordinal_type& i) {
                               Ss(k, i) = -Ys(k) * sumSpecSurf(i);
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
            Kokkos::atomic_fetch_add(&Ss(kspec, i), val);
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
            Kokkos::atomic_fetch_add(&Ss(kspec, i), val);
          }
        }
    });

    const ordinal_type n_surface_equations = kmcdSurf.nSpec - cstr.number_of_algebraic_constraints;
    const real_type site_density = kmcdSurf.sitedensity * real_type(10.0);
    const auto vsurfki=kmcdSurf.vsurfki;
    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(member, n_surface_equations),
                              [&](const ordinal_type& k) {
      Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(member, kmcdSurf.nReac),
                             [&](const ordinal_type& i) {
                               Cmat(k, i) = vsurfki(k,i)/site_density;
                             }); /* done loop over all reactions */
      });


    member.team_barrier();

#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("IsothermalTransientContStirredTankReactorSmatrix.team_invoke.test.out", "a+");
      fprintf(fs, ":: IsothermalTransientContStirredTankReactorSmatrix::team_invoke\n");
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
          fprintf(fs, "   i %d, j %d,  Smat %e, \n", i, j, Sg(i, j));
        }
      for (int i = 0; i < Nvars; ++i)
        for (int j = 0; j < kmcdSurf.nReac; j++) {
          fprintf(fs, "   i %d, j %d, Ss %e, \n", i, j, Ss(i, j));
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
    const real_type_2d_view_type& Smatrix,//gas terms
    // const real_type_2d_view_type& Cmat,
    // work
    /// workspace
    const WorkViewType& work,
    /// const input from kinetic model
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf,
    // const input for cstr
    const cstr_data_type& cstr)
  {
    using range_type = Kokkos::pair<ordinal_type, ordinal_type>;
    const ordinal_type total_n_reactions= kmcd.nReac + kmcdSurf.nReac;
    const ordinal_type total_n_equations = kmcd.nSpec + kmcdSurf.nSpec - cstr.number_of_algebraic_constraints;
    const real_type_2d_view_type Sg = Kokkos::subview(Smatrix,range_type(0, kmcd.nSpec),
                                            range_type(0, kmcd.nReac));
    const real_type_2d_view_type Ss = Kokkos::subview(Smatrix,range_type(0, kmcd.nSpec),
                                            range_type(kmcd.nReac,total_n_reactions ));
    const real_type_2d_view_type Sconv = Kokkos::subview(Smatrix,
                       range_type(0, kmcd.nSpec), range_type(total_n_reactions,total_n_reactions +1 ));
    const real_type_2d_view_type Cmat = Kokkos::subview(Smatrix,
                       range_type(kmcd.nSpec, total_n_equations), range_type(kmcd.nReac,total_n_reactions));
    auto w = (real_type*)work.data();
    auto sumSpecSurf = real_type_1d_view_type(w, kmcdSurf.nReac);
    w += kmcdSurf.nSpec;

    team_invoke_detail(member,
                       t,
                       Ys,
                       Zs,
                       density,
                       p,
                       Sg,
                       Ss,
                       Sconv,
                       Cmat,
                       sumSpecSurf,
                       kmcd,
                       kmcdSurf,
                       cstr);
  }
};

} // namespace Impl
} // namespace TChem

#endif
