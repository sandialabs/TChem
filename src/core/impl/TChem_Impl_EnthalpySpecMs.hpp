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


#ifndef __TCHEM_IMPL_ENTHALPYSPECMS_HPP__
#define __TCHEM_IMPL_ENTHALPYSPECMS_HPP__

#include "TChem_Impl_EnthalpySpecMl.hpp"
#include "TChem_Util.hpp"

namespace TChem {

namespace Impl {

struct EnthalpySpecMsFcn
{
  template<typename MemberType,
           typename RealType,
           typename RealType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const RealType& t, /// temperature
    /// output
    const RealType1DViewType& hi,
    /// work space
    const RealType1DViewType& cpks,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    EnthalpySpecMl::team_invoke(member, t, hi, cpks, kmcd);
    Kokkos::parallel_for(
      Kokkos::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& i) { hi(i) /= kmcd.sMass(i); });
#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("EnthalpySpecMs.team_invoke.test.out", "a+");
      fprintf(fs, ":: EnthalpySpecMs::team_invoke\n");
      fprintf(fs, ":::: input\n");
      fprintf(fs, "     nSpec %3d, t %e\n", kmcd.nSpec, t);
      fprintf(fs, ":::: output\n");
      for (int i = 0; i < kmcd.nSpec; ++i)
        fprintf(fs, "     i %3d, hi %e, cpks %e\n", i, hi(i), cpks(i));
    }
#endif
  }
};
using HspecMsFcn = EnthalpySpecMsFcn;     /// backward compatibility
using HspecMs = EnthalpySpecMsFcn;        /// backward compatibility
using EnthalpySpecMs = EnthalpySpecMsFcn; /// front interface
} // namespace Impl
} // namespace TChem

#endif
