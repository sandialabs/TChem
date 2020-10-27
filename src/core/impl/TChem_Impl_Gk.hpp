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
#ifndef __TCHEM_IMPL_GK_HPP__
#define __TCHEM_IMPL_GK_HPP__

#include "TChem_Impl_CpSpecMl.hpp"
#include "TChem_Impl_EnthalpySpecMl.hpp"
#include "TChem_Impl_Entropy0SpecMl.hpp"
#include "TChem_Util.hpp"

namespace TChem {
namespace Impl {

struct GkFcn
{
  template<typename MemberType,
           typename RealType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input temperature
    const real_type& t,
    /// output
    const RealType1DViewType& gk,
    const RealType1DViewType& hks,
    /// workspace
    const RealType1DViewType& cpks,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    const real_type t_1 = real_type(1) / t;
    const real_type tln = ats<real_type>::log(t);

    /// no need for barrier as all parallelized for kmcd.nSpec
    Entropy0SpecMl::team_invoke(member, t, gk, cpks, kmcd);

    EnthalpySpecMl::team_invoke(member, t, hks, cpks, kmcd);

    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, kmcd.nSpec),
                         [&](const ordinal_type& i) {
                           gk(i) = -tln + (gk(i) - hks(i) * t_1) / kmcd.Runiv;
                         });
#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("GkFcn.team_invoke.test.out", "a+");
      fprintf(fs, ":: Gk::team_invoke\n");
      fprintf(fs, ":::: input\n");
      fprintf(fs, "     nSpec %3d, t %e\n", kmcd.nSpec, t);
      fprintf(fs, ":::: output\n");
      for (int i = 0; i < int(gk.extent(0)); ++i)
        fprintf(fs, "     i %3d, gk %e\n", i, gk(i));
      for (int i = 0; i < int(hks.extent(0)); ++i)
        fprintf(fs, "     i %3d, hks %e\n", i, hks(i));
    }
#endif
  }
};
using Gk = GkFcn;

// Gas species in surface phase
struct GkFcnSurfGas
{
  template<typename MemberType,
           typename RealType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input temperature
    const real_type& t,
    /// output
    const RealType1DViewType& gk,
    const RealType1DViewType& hks,
    /// workspace
    const RealType1DViewType& cpks,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    const real_type t_1 = real_type(1) / t;
    const real_type tln = ats<real_type>::log(t);

    /// no need for barrier as all parallelized for kmcd.nSpec
    Entropy0SpecMl::team_invoke(member, t, gk, cpks, kmcd);

    EnthalpySpecMl::team_invoke(member, t, hks, cpks, kmcd);

    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, kmcd.nSpec),
                         [&](const ordinal_type& i) {
                           gk(i) = (gk(i) - hks(i) * t_1) / kmcd.Runiv;
                         });
#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("GkFcnSurfGas.team_invoke.test.out", "a+");
      fprintf(fs, ":: GkSurfGas::team_invoke\n");
      fprintf(fs, ":::: input\n");
      fprintf(fs, "     nSpec %3d, t %e\n", kmcd.nSpec, t);
      fprintf(fs, ":::: output\n");
      for (int i = 0; i < int(gk.extent(0)); ++i)
        fprintf(fs, "     i %3d, gk %e\n", i, gk(i));
      for (int i = 0; i < int(hks.extent(0)); ++i)
        fprintf(fs, "     i %3d, hks %e\n", i, hks(i));
    }
#endif
  }
};
using GkSurfGas = GkFcnSurfGas;

/// Derivative functions
struct GkFcnDerivative
{
  template<typename MemberType,
           typename RealType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input temperature
    const real_type& t,
    /// output
    const RealType1DViewType& gkp,
    const RealType1DViewType& hks,
    /// workspace
    const RealType1DViewType& cpks,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {

    /* done computing gkp=d(gk)/dT */
    const real_type t_1 = real_type(1) / t;

    /// no need for barrier as all parallelized for kmcd.nSpec
    Entropy0SpecMlDerivative::team_invoke(member, t, gkp, cpks, kmcd);

    EnthalpySpecMl::team_invoke(member, t, hks, cpks, kmcd);

    CpSpecMlDerivative::team_invoke(member, t, cpks, kmcd);

    Kokkos::parallel_for(
      Kokkos::TeamVectorRange(member, kmcd.nSpec), [&](const ordinal_type& i) {
        gkp(i) =
          -t_1 + (gkp(i) + hks(i) * t_1 * t_1 - cpks(i) * t_1) / kmcd.Runiv;
      });
#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("GkFcnDerivative.team_invoke.test.out", "a+");
      fprintf(fs, ":: GkDerivative::team_invoke\n");
      fprintf(fs, ":::: input\n");
      fprintf(fs, "     nSpec %3d, t %e\n", kmcd.nSpec, t);
      fprintf(fs, ":::: output\n");
      for (int i = 0; i < int(gkp.extent(0)); ++i)
        fprintf(fs,
                "     i %3d, gkp %e, hks %e, cpks %e\n",
                i,
                gkp(i),
                hks(i),
                cpks(i));
    }
#endif
  }
};
using GkDerivative = GkFcnDerivative;

} // namespace Impl
} // namespace TChem

#endif
