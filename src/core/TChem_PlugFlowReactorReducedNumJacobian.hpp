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
#ifndef __TCHEM_PLUGFLOWREACTORREDNUMJAC_HPP__
#define __TCHEM_PLUGFLOWREACTORREDNUMJAC_HPP__

#include "TChem_Impl_PlugFlowReactorReducedNumJacobian.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_PlugFlowReactorRHS.hpp"
#include "TChem_Util.hpp"

namespace TChem {

struct PlugFlowReactorReducedNumJacobian
{

  template<typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType,
           typename PlugFlowReactorConstDataType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf,
    const PlugFlowReactorConstDataType& pfrd)
  {
    return Impl::PlugFlowReactorReducedNumJacobian::
    getWorkSpaceSize(kmcd, kmcdSurf, pfrd);
  }

  static void runDeviceBatch( /// input
    const ordinal_type nBatch,
    /// input gas state
    const real_type_2d_view& state,
    /// surface state
    const real_type_2d_view& zSurf,
    // prf aditional variable
    const real_type_1d_view& velocity,
    /// output
    const real_type_3d_view& jac,
    const real_type_2d_view& fac,
    /// const data from kinetic model
    const KineticModelConstDataDevice& kmcd,
    /// const data from kinetic model surface
    const KineticSurfModelConstDataDevice& kmcdSurf,
    const pfr_data_type& pfrd);
};

} // namespace TChem

#endif
