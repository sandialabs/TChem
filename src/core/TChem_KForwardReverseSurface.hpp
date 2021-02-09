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


#ifndef __TCHEM_KFORWARD_REVERSE_SURF_HPP__
#define __TCHEM_KFORWARD_REVERSE_SURF_HPP__

#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"

namespace TChem {

struct KForwardReverseSurface
{
  template<typename KineticModelConstDataType,
           typename KineticModelConstSurfDataType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd,
    const KineticModelConstSurfDataType& kmcdSurf)
  {
    return (0 * kmcd.nSpec + 0 * kmcd.nReac + 0 * kmcdSurf.nSpec +
            0 * kmcdSurf.nReac);
  }

  // static void runHostBatch(/// input
  //                          const ordinal_type nBatch,
  //                          const real_type_2d_view_host &state,
  //                          /// input
  //                          const real_type_2d_view_host &zSurf,
  //                          /// output
  //                          const real_type_2d_view_host &omega,
  //                          const real_type_2d_view_host &omegaSurf,
  //                          /// const data from kinetic model
  //                          const KineticModelConstDataHost &kmcd,
  //                          /// const data from kinetic model surface
  //                          const KineticSurfModelConstDataHost &kmcdSurf);

  static void runDeviceBatch( /// input
    const ordinal_type nBatch,
    /// input
    const real_type_2d_view& state,
    const real_type_2d_view& gk,
    const real_type_2d_view& gkSurf,
    /// output
    const real_type_2d_view& kfor,
    const real_type_2d_view& krev,
    /// const data from kinetic model
    const KineticModelConstDataDevice& kmcd,
    /// const data from kinetic model surface
    const KineticSurfModelConstDataDevice& kmcdSurf);
};

} // namespace TChem

#endif
