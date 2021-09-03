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
#ifndef __TCHEM_PLUGFLOWREACTORNUMJAC2_HPP__
#define __TCHEM_PLUGFLOWREACTORNUMJAC2_HPP__

#include "TChem_Impl_PlugFlowReactorNumJacobian.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_PlugFlowReactorRHS.hpp"
#include "TChem_PlugFlowReactor.hpp"
#include "TChem_Util.hpp"

namespace TChem {

struct PlugFlowReactorNumJacobian
{

  using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;
  using device_type      = typename Tines::UseThisDevice<exec_space>::type;

  using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;
  using real_type_2d_view_type = Tines::value_type_2d_view<real_type,device_type>;
  using real_type_3d_view_type = Tines::value_type_3d_view<real_type,device_type>;

  using real_type_1d_view_host_type = Tines::value_type_1d_view<real_type,host_device_type>;
  using real_type_2d_view_host_type = Tines::value_type_2d_view<real_type,host_device_type>;
  using real_type_3d_view_host_type = Tines::value_type_3d_view<real_type,host_device_type>;

  using kinetic_model_type = KineticModelConstData<device_type>;
  using kinetic_surf_model_type = KineticSurfModelConstData<device_type>;

  using kinetic_model_host_type = KineticModelConstData<host_device_type>;
  using kinetic_surf_model_host_type = KineticSurfModelConstData<host_device_type>;

  template<typename DeviceType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstData<DeviceType >& kmcd,
    const KineticSurfModelConstData<DeviceType>& kmcdSurf)
  {
    return Impl::PlugFlowReactorNumJacobian<real_type,DeviceType>::
    getWorkSpaceSize(kmcd, kmcdSurf);
  }

  static void runDeviceBatch( /// input
    const ordinal_type& team_size,
    const ordinal_type& vector_size,
    const ordinal_type nBatch,
    /// input gas state
    const real_type_2d_view_type& state,
    /// surface state
    const real_type_2d_view_type& site_fraction,
    // prf aditional variable
    const real_type_1d_view_type& velocity,
    /// output
    const real_type_3d_view_type& jac,
    const real_type_2d_view_type& fac,
    /// const data from kinetic model
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf,
    const PlugFlowReactorData& pfrd);
  //
  static void runDeviceBatch( /// input
    typename UseThisTeamPolicy<exec_space>::type& policy,
    /// input gas state
    const real_type_2d_view_type& state,
    /// surface state
    const real_type_2d_view_type& site_fraction,
    // prf aditional variable
    const real_type_1d_view_type& velocity,
    /// output
    const real_type_3d_view_type& jac,
    const real_type_2d_view_type& fac,
    /// const data from kinetic model
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf,
    //const data from pfr reactor
    const PlugFlowReactorData& pfrd);
};




} // namespace TChem

#endif
