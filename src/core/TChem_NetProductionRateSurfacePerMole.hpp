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
#ifndef __TCHEM_REACTION_RATES_SURFMOLE_HPP__
#define __TCHEM_REACTION_RATES_SURFMOLE_HPP__

#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"
// #include "TChem_KineticModelSurfData.hpp"

namespace TChem {

struct NetProductionRateSurfacePerMole
{
  template<typename KineticModelConstDataType,
           typename KineticModelConstSurfDataType>
  KOKKOS_INLINE_FUNCTION
  static ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd,
    const KineticModelConstSurfDataType& kmcdSurf)
  {
    return (4 * kmcd.nSpec + 0 * kmcd.nReac + 4 * kmcdSurf.nSpec +
            4 * kmcdSurf.nReac);
  }

  static void runHostBatch( /// input
    const ordinal_type nBatch,
    const real_type_2d_view_host& state,
    /// input
    const real_type_2d_view_host& zSurf,
    /// output
    const real_type_2d_view_host& omega,
    const real_type_2d_view_host& omegaSurf,
    /// const data from kinetic model
    const KineticModelConstDataHost& kmcd,
    /// const data from kinetic model surface
    const KineticSurfModelConstDataHost& kmcdSurf);

  static void runDeviceBatch( /// input
    const ordinal_type nBatch,
    const real_type_2d_view& state,
    /// input
    const real_type_2d_view& zSurf,
    /// output
    const real_type_2d_view& omega,
    const real_type_2d_view& omegaSurf,
    /// const data from kinetic model
    const KineticModelConstDataDevice& kmcd,
    /// const data from kinetic model surface
    const KineticSurfModelConstDataDevice& kmcdSurf);
};

} // namespace TChem

#endif
