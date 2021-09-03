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

#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"

namespace TChem {
namespace Impl {
///
/// \sum_{reaction} {\nu_{ki} g_{k}}
///
template<typename ValueType, typename DeviceType>
struct SumNuGk
{
  using value_type = ValueType;
  using device_type = DeviceType;
  using scalar_type = typename ats<value_type>::scalar_type;

  using real_type = scalar_type;
  /// sacado is value type
  using value_type_1d_view_type = Tines::value_type_1d_view<value_type,device_type>;
  using kinetic_model_type= KineticModelConstData<device_type>;
  using kinetic_surf_model_type = KineticSurfModelConstData<device_type>;

  KOKKOS_INLINE_FUNCTION static value_type serial_invoke( /// input
    const ordinal_type& i,
    const value_type_1d_view_type& gk,
    /// const input from kinetic model
    const kinetic_model_type& kmcd)
  {
    value_type sumNuGk(0);
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

  KOKKOS_INLINE_FUNCTION static value_type serial_invoke( /// input
    const ordinal_type& i,
    const value_type_1d_view_type& gk,
    const value_type_1d_view_type& gkSurf,
    /// const input from kinetic model
    const kinetic_surf_model_type& kmcdSurf)
  {
    value_type sumNuGk(0);
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
