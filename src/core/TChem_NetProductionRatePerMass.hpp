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
#ifndef __TCHEM_REACTION_RATES_HPP__
#define __TCHEM_REACTION_RATES_HPP__

#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"

namespace TChem {

struct NetProductionRatePerMass
{
  template<typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION
  static ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd)
  {
    return (4 * kmcd.nSpec + 7 * kmcd.nReac);
  }

  static void runHostBatch( /// input
    const ordinal_type nBatch,
    const real_type_2d_view_host& state,
    /// output
    const real_type_2d_view_host& omega,
    /// const data from kinetic model
    const KineticModelConstDataHost& kmcd);

  static void runDeviceBatch( /// input
    const ordinal_type nBatch,
    const real_type_2d_view& state,
    /// output
    const real_type_2d_view& omega,
    /// const data from kinetic model
    const KineticModelConstDataDevice& kmcd);
};

} // namespace TChem

#endif
