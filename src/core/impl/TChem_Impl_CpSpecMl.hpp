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


#ifndef __TCHEM_IMPL_CPSPECML_HPP__
#define __TCHEM_IMPL_CPSPECML_HPP__

#include "TChem_Util.hpp"

namespace TChem {
namespace Impl {

/// all impl space evaluate values (nspecs) per a single temperature input
struct CpSpecMlFcn
{
  template<typename MemberType,
           typename RealType,
           typename RealType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const RealType& t,
    /// output
    const RealType1DViewType& cpi,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    const auto tLoc = getValueInRangev2(kmcd.TthrmMin, kmcd.TthrmMax, t);
    Kokkos::parallel_for(
      Kokkos::TeamVectorRange(member, kmcd.nSpec), [&](const ordinal_type& i) {
        const ordinal_type ipol = tLoc > kmcd.Tmi(i);
        // this assumes nNASAinter_ = 2, nCpCoef_ = 5 (confirm this)
        // icpst = i*7*2+ipol*7 ;
        cpi(i) = (kmcd.cppol(i, ipol, 0) +
                  tLoc * (kmcd.cppol(i, ipol, 1) +
                          tLoc * (kmcd.cppol(i, ipol, 2) +
                                  tLoc * (kmcd.cppol(i, ipol, 3) +
                                          tLoc * kmcd.cppol(i, ipol, 4)))));
        cpi(i) *= kmcd.Runiv;
      });
#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("CpSpecMl.team_invoke.test.out", "a+");
      fprintf(fs, ":: CpSpecMl::team_invoke\n");
      fprintf(fs, ":::: input\n");
      fprintf(fs, "     nSpec %3d, t %e\n", kmcd.nSpec, t);
      fprintf(fs, ":::: output\n");
      for (int i = 0; i < kmcd.nSpec; ++i)
        fprintf(fs, "     i %3d, cpi %e\n", i, cpi(i));
    }
#endif
  }
};
using CpSpecMl = CpSpecMlFcn;

struct CpSpecMlFcnDerivative
{
  template<typename MemberType,
           typename RealType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const real_type& t,
    /// output
    const RealType1DViewType& cpi,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    const real_type zero(0), two(2), three(3), four(4);
    const real_type tLoc = getValueInRange(kmcd.TthrmMin, kmcd.TthrmMax, t);
    const real_type delT = t - tLoc;

    if (ats<real_type>::abs(delT) > REACBALANCE()) {
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, kmcd.nSpec),
                           [&](const ordinal_type& i) { cpi(i) = zero; });
    } else {
      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, kmcd.nSpec),
        [&](const ordinal_type& i) {
          const ordinal_type ipol = tLoc > kmcd.Tmi(i);
          // this assumes nNASAinter_ = 2, nCpCoef_ = 5 (confirm this)
          // icpst = i*7*2+ipol*7 ;
          cpi(i) = (kmcd.cppol(i, ipol, 1) +
                    tLoc * (two * kmcd.cppol(i, ipol, 2) +
                            tLoc * (three * kmcd.cppol(i, ipol, 3) +
                                    tLoc * four * kmcd.cppol(i, ipol, 4))));
          cpi(i) *= kmcd.Runiv;
        });
    }
#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("CpSpecMlDerivative.team_invoke.test.out", "a+");
      fprintf(fs, ":: CpSpecMlDerivative::team_invoke\n");
      fprintf(fs, ":::: input\n");
      fprintf(fs, "     nSpec %3d, t %e\n", kmcd.nSpec, t);
      fprintf(fs, ":::: output\n");
      for (int i = 0; i < kmcd.nSpec; ++i)
        fprintf(fs, "     i %3d, cpi %e\n", i, cpi(i));
    }
#endif
  }
};
using CpSpecMlDerivative = CpSpecMlFcnDerivative;

} // namespace Impl
} // namespace TChem

#endif
