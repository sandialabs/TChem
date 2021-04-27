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
#ifndef __TCHEM_IMPL_SMATRIX_HPP__
#define __TCHEM_IMPL_SMATRIX_HPP__

#include "TChem_Impl_CpMixMs.hpp"
#include "TChem_Impl_EnthalpySpecMs.hpp"
#include "TChem_Impl_RhoMixMs.hpp"
#include "TChem_Util.hpp"

/**
   \param scal : array of Nspec+1 doubles \f$(T,Y_1,Y_2,...,Y_{N_{spec}})\f$
0	        temperature T [K], mass fractions Y []
   \param Nvars : no. of variables \f$ N_{vars}= N_{spec}+1\f$
   \return Smat : array of \f$(N_{spec}+1)\times 2N_{reac}\f$ holding
                  the S components in column major format
*/

namespace TChem {
namespace Impl {

struct Smatrix
{
  template<typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd)
  {
    return 2 * kmcd.nSpec;
  }

  template<typename MemberType,
           typename RealType1DViewType,
           typename RealType2DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    /// input
    const real_type& t,
    const real_type& p,
    const RealType1DViewType& Ys, /// (kmcd.nSpec)
    /// output
    const RealType2DViewType& Smat, /// (1)
    /// workspace
    const RealType1DViewType& hks,
    const RealType1DViewType& cpks,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    const real_type zero(0), one(1);

    const ordinal_type Nvars = kmcd.nSpec + one;

    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, kmcd.nReac),
                         [=](const ordinal_type& i) {
                           for (ordinal_type j = 0; j < Nvars; ++j) {
                             Smat(j, i) = zero;
                           }
                         }); /* done loop over all reactions */

    /// 3. compute density, cpmix
    const real_type rhomix = RhoMixMs::team_invoke(member, t, p, Ys, kmcd);
    const real_type cpmix = CpMixMs::team_invoke(member, t, Ys, cpks, kmcd);

    /// 4. compute species enthalies
    EnthalpySpecMs ::team_invoke(member, t, hks, cpks, kmcd);

    const real_type ConEnergy = -1. / (cpmix * rhomix);
    member.team_barrier();
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
                             Smat(kspec + 1, i) =
                               kmcd.sMass(kspec) * kmcd.reacNuki(i, j) / rhomix;
                             // Kokkos::atomic_fetch_add(&Smat(kspec+ 1,i),
                             // val);
                           });

      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, kmcd.reacNprod(i)),
        [=](const ordinal_type& j) {
          const ordinal_type kspec = kmcd.reacSidx(i, j + joff);
          Smat(kspec + 1, i) =
            kmcd.sMass(kspec) * kmcd.reacNuki(i, j + joff) / rhomix;
          // Kokkos::atomic_fetch_add(&Smat(kspec+ 1,i), val);
        });
    }

    /* add contributions from real stoichiometric coefficients if any */
  }

  template<typename MemberType,
           typename WorkViewType,
           typename RealType1DViewType,
           typename RealType2DViewType,
           typename KineticModelConstDataType>
  KOKKOS_FORCEINLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const real_type& t,
    const real_type& p,
    const RealType1DViewType& Ys, /// (kmcd.nSpec)
    /// output
    const RealType2DViewType& Smat,
    /// workspace
    const WorkViewType& work,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    // const real_type zero(0);

    ///
    /// workspace needed gk, hks, kfor, krev
    ///
    auto w = (real_type*)work.data();

    auto hks = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto cpks = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;

    team_invoke_detail(member,
                       t,
                       p,
                       Ys,
                       Smat,
                       /// workspace
                       hks,
                       cpks,
                       kmcd);
  }
};

} // namespace Impl
} // namespace TChem

#endif
