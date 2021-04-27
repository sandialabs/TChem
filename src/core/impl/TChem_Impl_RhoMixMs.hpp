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
#ifndef __TCHEM_IMPL_RHOMIXMS_HPP__
#define __TCHEM_IMPL_RHOMIXMS_HPP__

#include "TChem_Util.hpp"

namespace TChem {

namespace Impl {

/// Computes density based on temperature and species mole fractions using the
/// equation of state (getRhoMixMs)
///    \param t : temperature
///    \param p : pressure
///    \param Ys : array of mass fractions Y
///    \return mixture density [kg/m<sup>3</sup>]
struct RhoMixMs
{
  template<typename MemberType,
           typename RealType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static real_type team_invoke(
    const MemberType& member,
    /// input
    const real_type& t,           /// temperature
    const real_type& p,           /// pressure
    const RealType1DViewType& Ys, /// mole fractions
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    real_type Ysum(0);
    Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& i, real_type& update) {
        update += Ys(i) / kmcd.sMass(i);
      },
      Ysum);

    const real_type r_val = p / (kmcd.Runiv * Ysum * t);
#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("RhoMixMs.team_invoke.test.out", "a+");
      fprintf(fs, ":: RhoMixMs::team_invoke\n");
      fprintf(fs, ":::: input\n");
      fprintf(fs,
              "     nSpec %3d, t %e, p %e, kmcd.Runiv %e\n",
              kmcd.nSpec,
              t,
              p,
              kmcd.Runiv);
      for (int i = 0; i < kmcd.nSpec; ++i)
        fprintf(fs, "     i %3d, Ys %e, sMass %e\n", i, Ys(i), kmcd.sMass(i));
      fprintf(fs, ":::: output\n");
      fprintf(fs, "     rhomix %e\n", r_val);
    }
#endif
    return r_val;
  }
};

} // namespace Impl
} // namespace TChem

#endif
