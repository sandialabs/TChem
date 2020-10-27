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
#ifndef __TCHEM_SURFACERHS_HPP__
#define __TCHEM_SURFACERHS_HPP__

#include "TChem_Impl_SurfaceRHS.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"

namespace TChem {

struct SurfaceRHS
{
  template<typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf)
  {
    // work for surface reactions
    const ordinal_type per_team_extent =
      Impl::ReactionRatesSurface::getWorkSpaceSize(kmcd, kmcdSurf);

    return (per_team_extent + kmcd.nSpec + kmcdSurf.nSpec);
  }

  static void runHostBatch( /// input
    const ordinal_type nBatch,
    const real_type_2d_view_host& state,
    /// input
    const real_type_2d_view_host& zSurf,
    // prf aditional variable
    const real_type_2d_view_host& velocity,
    /// output
    const real_type_2d_view_host& rhs,
    /// const data from kinetic model
    const KineticModelConstDataHost& kmcd,
    /// const data from kinetic model surface
    const KineticSurfModelConstDataHost& kmcdSurf);

  static void runDeviceBatch( /// input
    const ordinal_type nBatch,
    /// input gas state
    const real_type_2d_view& state,
    /// surface state
    const real_type_2d_view& zSurf,
    /// output
    const real_type_2d_view& rhs,
    /// const data from kinetic model
    const KineticModelConstDataDevice& kmcd,
    /// const data from kinetic model surface
    const KineticSurfModelConstDataDevice& kmcdSurf);
};

} // namespace TChem

#endif
