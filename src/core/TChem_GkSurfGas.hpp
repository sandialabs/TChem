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
#ifndef __TCHEM_GK_SURF_HPP__
#define __TCHEM_GK_SURF_HPP__

#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"
// #include "TChem_KineticModelSurfData.hpp"

namespace TChem {

struct GkSurfGas
{
  template<typename KineticModelConstDataType,
           typename KineticModelConstSurfDataType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd,
    const KineticModelConstSurfDataType& kmcdSurf)
  {
    return (1 * kmcd.nSpec + 0 * kmcd.nReac + 1 * kmcdSurf.nSpec +
            0 * kmcdSurf.nReac);
  }

  static void runHostBatch( /// input
    const ordinal_type nBatch,
    const real_type_2d_view_host& state,
    /// output
    const real_type_2d_view_host& gk,
    const real_type_2d_view_host& gkSurf,
    const real_type_2d_view_host& hks,
    const real_type_2d_view_host& hksSurf,
    /// const data from kinetic model
    const KineticModelConstDataHost& kmcd,
    /// const data from kinetic model surface
    const KineticSurfModelConstDataHost& kmcdSurf);

  static void runDeviceBatch( /// input
    const ordinal_type nBatch,
    const real_type_2d_view& state,
    /// output
    const real_type_2d_view& gk,
    const real_type_2d_view& gkSurf,
    const real_type_2d_view& hks,
    const real_type_2d_view& hksSurf,
    /// const data from kinetic model
    const KineticModelConstDataDevice& kmcd,
    /// const data from kinetic model surface
    const KineticSurfModelConstDataDevice& kmcdSurf);
};

} // namespace TChem

#endif
