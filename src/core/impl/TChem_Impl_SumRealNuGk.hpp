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


#ifndef __TCHEM_IMPL_SUMREALNUGK_HPP__
#define __TCHEM_IMPL_SUMREALNUGK_HPP__

#include "TChem_Util.hpp"

namespace TChem {
namespace Impl {

///
/// ...
///
struct SumRealNuGk
{
  template<typename RealType,
           typename RealType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static RealType serial_invoke( /// input
    const RealType& dummy, 
    const ordinal_type& i,
    const ordinal_type& ir,
    const RealType1DViewType& gk,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    RealType sumNuGk(0);
    for (ordinal_type j = 0; j < kmcd.reacNreac(i); ++j) {
      const ordinal_type kspec = kmcd.reacSidx(i, j);
      sumNuGk += kmcd.reacRealNuki(ir, j) * gk(kspec);
    } /* done for loop over reactants */

    const ordinal_type joff_nuki = kmcd.reacNreac(i);
    const ordinal_type joff_sidx = kmcd.reacSidx.extent(1) / 2;
    for (ordinal_type j = 0; j < kmcd.reacNprod(i); ++j) {
      const ordinal_type kspec = kmcd.reacSidx(i, j + joff_sidx);
      sumNuGk += kmcd.reacRealNuki(i, j + joff_nuki) * gk(kspec);
    } /* done loop over products */
    return (sumNuGk);
  }
};

} // namespace Impl
} // namespace TChem

#endif
