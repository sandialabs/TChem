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
#ifndef __TCHEM_PLUGFLOWREACTORSMAT_HPP__
#define __TCHEM_PLUGFLOWREACTORSMAT_HPP__

#include "TChem_Impl_PlugFlowReactorSmat.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_PlugFlowReactorRHS.hpp"
#include "TChem_Util.hpp"

namespace TChem {

struct PlugFlowReactorSmat
{

  template<typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf)
  {
    return (2 * kmcd.nSpec + kmcdSurf.nReac);
  }

  // static void runHostBatch(/// input
  //                          const ordinal_type nBatch,
  //                          const real_type_2d_view_host &state,
  //                          /// input
  //                          const real_type_2d_view_host &zSurf,
  //                          // prf aditional variable
  //                          const real_type_2d_view_host &velocity,
  //                          /// output
  //                          const real_type_3d_view_host &Gu,
  //                          const real_type_3d_view_host &Fu,
  //                          const real_type_3d_view_host &Gv,
  //                          const real_type_3d_view_host &Fv,
  //
  //                          /// const data from kinetic model
  //                          const KineticModelConstDataHost &kmcd,
  //                          /// const data from kinetic model surface
  //                          const KineticSurfModelConstDataHost &kmcdSurf);

  static void runDeviceBatch( /// input
    const ordinal_type nBatch,
    /// input gas state
    const real_type_2d_view& state,
    /// surface state
    const real_type_2d_view& zSurf,
    // prf aditional variable
    const real_type_1d_view& velocity,
    /// output
    const real_type_3d_view& Smat,
    const real_type_3d_view& Ssmat,
    /// const data from kinetic model
    const KineticModelConstDataDevice& kmcd,
    /// const data from kinetic model surface
    const KineticSurfModelConstDataDevice& kmcdSurf);
};

} // namespace TChem

#endif
