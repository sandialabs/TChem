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
  using device_type      = typename Tines::UseThisDevice<exec_space>::type;

  using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;
  using real_type_2d_view_type = Tines::value_type_2d_view<real_type,device_type>;

  using kinetic_model_type = KineticModelConstData<device_type>;
  using kinetic_surf_model_type = KineticSurfModelConstData<device_type>;



  template<typename DeviceType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstData<DeviceType >& kmcd,
    const KineticSurfModelConstData<DeviceType>& kmcdSurf)
  {
    // work for surface reactions
    using device_type = DeviceType;
    using ReactionRatesSurface = Impl::ReactionRatesSurface<real_type,device_type>;

    const ordinal_type per_team_extent =
      ReactionRatesSurface::getWorkSpaceSize(kmcd, kmcdSurf);

    return (per_team_extent + kmcd.nSpec + kmcdSurf.nSpec);
  }

  static void runDeviceBatch( /// input
    const ordinal_type nBatch,
    /// input gas state
    const real_type_2d_view_type& state,
    /// surface state
    const real_type_2d_view_type& zSurf,
    /// output
    const real_type_2d_view_type& rhs,
    /// const data from kinetic model
    const kinetic_model_type& kmcd,
    /// const data from kinetic model surface
    const kinetic_surf_model_type& kmcdSurf);
};

} // namespace TChem

#endif
