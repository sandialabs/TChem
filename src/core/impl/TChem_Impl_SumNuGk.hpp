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
#ifndef __TCHEM_IMPL_SUMNUGK_HPP__
#define __TCHEM_IMPL_SUMNUGK_HPP__

#include "TChem_Util.hpp"

namespace TChem {
namespace Impl {
///
/// \sum_{reaction} {\nu_{ki} g_{k}}
///
struct SumNuGk
{
  template<typename RealType,
           typename RealType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static RealType serial_invoke( /// input
    const RealType& dummy,
    const ordinal_type& i,
    const RealType1DViewType& gk,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    RealType sumNuGk(0);
    for (ordinal_type j = 0; j < kmcd.reacNreac(i); ++j) {
      const ordinal_type kspec = kmcd.reacSidx(i, j);
      sumNuGk += kmcd.reacNuki(i, j) * gk(kspec);
    } /* done for loop over reactants */

    const ordinal_type joff = kmcd.reacSidx.extent(1) / 2;
    for (ordinal_type j = 0; j < kmcd.reacNprod(i); ++j) {
      const ordinal_type kspec = kmcd.reacSidx(i, j + joff);
      sumNuGk += kmcd.reacNuki(i, j + joff) * gk(kspec);
    } /* done loop over products */
    return (sumNuGk);
  }

  template<typename RealType1DViewType,
           typename KineticSurfModelConstDataType>
  KOKKOS_INLINE_FUNCTION static real_type serial_invoke( /// input
    const ordinal_type& i,
    const RealType1DViewType& gk,
    const RealType1DViewType& gkSurf,
    /// const input from kinetic model
    const KineticSurfModelConstDataType& kmcdSurf)
  {
    real_type sumNuGk(0);
    for (ordinal_type j = 0; j < kmcdSurf.reacNreac(i); ++j) {
      const ordinal_type kspec = kmcdSurf.reacSidx(i, j);
      if (kmcdSurf.reacSsrf(i, j) == 1) { // surfaces
        sumNuGk += kmcdSurf.reacNuki(i, j) * gkSurf(kspec);
      } else if (kmcdSurf.reacSsrf(i, j) == 0) { // gas
        sumNuGk += kmcdSurf.reacNuki(i, j) * gk(kspec);
      }
    } /* done for loop over reactants */

    const ordinal_type joff = kmcdSurf.maxSpecInReac / 2;
    for (ordinal_type j = 0; j < kmcdSurf.reacNprod(i); ++j) {
      const ordinal_type kspec = kmcdSurf.reacSidx(i, j + joff);
      if (kmcdSurf.reacSsrf(i, j + joff) == 1) { // surfaces
        sumNuGk += kmcdSurf.reacNuki(i, j + joff) * gkSurf(kspec);
      } else if (kmcdSurf.reacSsrf(i, j + joff) == 0) { // gas
        sumNuGk += kmcdSurf.reacNuki(i, j + joff) * gk(kspec);
      }
    } /* done loop over products */
    return (sumNuGk);
  }
};

} // namespace Impl
} // namespace TChem

#endif
