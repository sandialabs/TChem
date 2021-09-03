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
#ifndef __TCHEM_KFORWARD_REVERSE_SURF_HPP__
#define __TCHEM_KFORWARD_REVERSE_SURF_HPP__

#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"

namespace TChem {

struct KForwardReverseSurface
{
  using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;
  using device_type      = typename Tines::UseThisDevice<exec_space>::type;

  using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;
  using real_type_2d_view_type = Tines::value_type_2d_view<real_type,device_type>;

  using real_type_1d_view_host_type = Tines::value_type_1d_view<real_type,host_device_type>;
  using real_type_2d_view_host_type = Tines::value_type_2d_view<real_type,host_device_type>;

  using kinetic_model_type = KineticModelConstData<device_type>;
  using kinetic_model_host_type = KineticModelConstData<host_device_type>;

  using kinetic_surf_model_type = KineticSurfModelConstData<device_type>;
  using kinetic_surf_model_host_type = KineticSurfModelConstData<host_device_type>;

  template<typename DeviceType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstData<DeviceType>& kmcd,
    const KineticSurfModelConstData<DeviceType>& kmcdSurf)
  {
    return (0);
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
    const real_type_2d_view_type& state,
    const real_type_2d_view_type& gk,
    const real_type_2d_view_type& gkSurf,
    /// output
    const real_type_2d_view_type& kfor,
    const real_type_2d_view_type& krev,
    /// const data from kinetic model
    const kinetic_model_type& kmcd,
    /// const data from kinetic model surface
    const kinetic_surf_model_type& kmcdSurf);
};

} // namespace TChem

#endif
