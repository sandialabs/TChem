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
#ifndef __TCHEM_IMPL_ENTROPY0SPECML_HPP__
#define __TCHEM_IMPL_ENTROPY0SPECML_HPP__

#include "TChem_Impl_CpSpecMl.hpp"
#include "TChem_Impl_CpSpecMs.hpp"
#include "TChem_Util.hpp"

namespace TChem {
namespace Impl {

struct Entropy0SpecMlFcn
{
  template<typename MemberType,
           typename RealType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const real_type& t,
    /// output (nspec)
    const RealType1DViewType& s0i,
    /// workspace
    const RealType1DViewType& cpks,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    const real_type one[3] = { 0.5, (1.0 / 3.0), 0.25 };
    const real_type tLoc = getValueInRange(kmcd.TthrmMin, kmcd.TthrmMax, t);
    const real_type delT = t - tLoc;
    const real_type tln = ats<real_type>::log(tLoc);
    Kokkos::parallel_for(
      Kokkos::TeamVectorRange(member, kmcd.nSpec), [&](const ordinal_type& i) {
        const ordinal_type ipol = tLoc > kmcd.Tmi(i);
        // this assumes nNASAinter_ = 2, nCpCoef_ = 5 (confirm this)
        // icpst = i*7*2+ipol*7 ;
        s0i(i) =
          kmcd.cppol(i, ipol, 0) * tln +
          tLoc * (kmcd.cppol(i, ipol, 1) +
                  tLoc * (kmcd.cppol(i, ipol, 2) * one[0] +
                          tLoc * (kmcd.cppol(i, ipol, 3) * one[1] +
                                  tLoc * kmcd.cppol(i, ipol, 4) * one[2]))) +
          kmcd.cppol(i, ipol, 6);
        s0i(i) *= kmcd.Runiv;
      });

    /* Check if temperature outside bounds */
    if (ats<real_type>::abs(delT) > REACBALANCE) {
      CpSpecMl::team_invoke(member, tLoc, cpks, kmcd);
      const real_type t_tLoc = t / tLoc;
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, kmcd.nSpec),
                           [&](const ordinal_type& i) {
                             s0i(i) += cpks(i) * ats<real_type>::log(t_tLoc);
                           });
    }
  }
};
using Entr0SpecMlFcn = Entropy0SpecMlFcn; /// backward compatibility
using Entropy0SpecMl = Entropy0SpecMlFcn; /// front interface

struct Entropy0SpecMlFcnDerivative
{
  template<typename MemberType,
           typename RealType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const real_type& t,
    /// output (nspec)
    const RealType1DViewType& ei,
    /// workspace (cpks)
    const RealType1DViewType& cpks,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    const real_type tLoc = getValueInRange(kmcd.TthrmMin, kmcd.TthrmMax, t);
    const real_type delT = t - tLoc;
    const real_type tln = ats<real_type>::log(tLoc);

    if (ats<real_type>::abs(delT) > REACBALANCE) {
      CpSpecMs::team_invoke(member, t, cpks, kmcd);
      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, kmcd.nSpec),
        [&](const ordinal_type& i) { ei(i) = cpks(i) / tLoc; });
    } else {
      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, kmcd.nSpec),
        [&](const ordinal_type& i) {
          const ordinal_type ipol = tLoc > kmcd.Tmi(i);
          // this assumes nNASAinter_ = 2, nCpCoef_ = 5 (confirm this)
          // icpst = i*7*2+ipol*7 ;
          ei(i) = (kmcd.cppol(i, ipol, 0) / tLoc + kmcd.cppol(i, ipol, 1) +
                   tLoc * (kmcd.cppol(i, ipol, 2) +
                           tLoc * (kmcd.cppol(i, ipol, 3) +
                                   tLoc * kmcd.cppol(i, ipol, 4))));
          ei(i) *= kmcd.Runiv;
        });
    }
  }
};
using Entropy0SpecMlDerivative = Entropy0SpecMlFcnDerivative;

} // namespace Impl
} // namespace TChem

#endif
