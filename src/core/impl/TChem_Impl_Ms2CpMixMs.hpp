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
#ifndef __TCHEM_IMPL_MS2CPMIXMS_HPP__
#define __TCHEM_IMPL_MS2CPMIXMS_HPP__

#include "TChem_Util.hpp"

namespace TChem {

namespace Impl {

struct Ms2CpMixMs
{
  template<typename MemberType,
           typename RealType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static real_type team_invoke(
    const MemberType& member,
    /// input
    const real_type& t, /// temperature
    const RealType1DViewType& Ys,
    const RealType1DViewType& cpks,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    CpSpecMs::team_invoke(member, t, Ys, cpks, kmcd);
    real_type cpmix(0);
    Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(member, kmcd.nSpec),
      [&](const ordinal_type& i, real_type& update) {
        update += Ys(i) * cpks(i);
      },
      cpmix);
    return cpmix;
  }
};

} // namespace Impl
} // namespace TChem

#endif
