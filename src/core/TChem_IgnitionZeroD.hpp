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
#ifndef __TCHEM_IGNITION_ZEROD_HPP__
#define __TCHEM_IGNITION_ZEROD_HPP__

#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"

#include "TChem_Impl_IgnitionZeroD.hpp"

namespace TChem {

struct IgnitionZeroD
{
  template<typename DeviceType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstData<DeviceType>& kmcd)
  {
    using device_type = DeviceType;
    using problem_type = Impl::IgnitionZeroD_Problem<real_type, device_type>;
    const ordinal_type m = problem_type::getNumberOfEquations(kmcd) + ordinal_type(1);

    ordinal_type work_size(0);
#if defined(TCHEM_ENABLE_SACADO_JACOBIAN_IGNITION_ZERO_D_REACTOR)
    if (m < 16) {
      using value_type = Sacado::Fad::SLFad<real_type,16>;
      work_size = Impl::IgnitionZeroD<value_type, device_type>::getWorkSpaceSize(kmcd);
    } else if  (m < 32) {
      using value_type = Sacado::Fad::SLFad<real_type,32>;
      work_size = Impl::IgnitionZeroD<value_type, device_type>::getWorkSpaceSize(kmcd);
    } else if  (m < 64) {
      using value_type = Sacado::Fad::SLFad<real_type,64>;
      work_size = Impl::IgnitionZeroD<value_type, device_type>::getWorkSpaceSize(kmcd);
    } else if  (m < 128) {
      using value_type = Sacado::Fad::SLFad<real_type,128>;
      work_size = Impl::IgnitionZeroD<value_type, device_type>::getWorkSpaceSize(kmcd);
    } else if  (m < 256) {
      using value_type = Sacado::Fad::SLFad<real_type,256>;
      work_size = Impl::IgnitionZeroD<value_type, device_type>::getWorkSpaceSize(kmcd);
    } else if  (m < 512) {
      using value_type = Sacado::Fad::SLFad<real_type,512>;
      work_size = Impl::IgnitionZeroD<value_type, device_type>::getWorkSpaceSize(kmcd);
    } else if (m < 1024){
      using value_type = Sacado::Fad::SLFad<real_type,1024>;
      work_size = Impl::IgnitionZeroD<value_type, device_type>::getWorkSpaceSize(kmcd);
    } else{
      TCHEM_CHECK_ERROR(0,
                        "Error: Number of equations is bigger than size of sacado fad type");
    }
#else
    {
      work_size = Impl::IgnitionZeroD<real_type, device_type>::getWorkSpaceSize(kmcd);
    }
#endif
    return work_size + m;
  }

  /// tadv - an input structure for time marching
  /// state (nSpec+3) - initial condition of the state vector
  /// work - work space sized by getWorkSpaceSize
  /// t_out - time when this code exits
  /// state_out - final condition of the state vector (the same input state can
  /// be overwritten) kmcd - const data for kinetic model
  static void runDeviceBatch( /// thread block size
    typename UseThisTeamPolicy<exec_space>::type& policy,
    /// global tolerence parameters that governs all samples
    const real_type_1d_view& tol_newton,
    const real_type_2d_view& tol_time,
    /// sample specific input
    const real_type_2d_view& fac,
    const time_advance_type_1d_view& tadv,
    const real_type_2d_view& state,
    /// output
    const real_type_1d_view& t_out,
    const real_type_1d_view& dt_out,
    const real_type_2d_view& state_out,
    /// const data from kinetic model
    const Tines::value_type_1d_view<KineticModelConstData<interf_device_type>,interf_device_type>& kmcds);

  static void runHostBatch( /// input
    typename UseThisTeamPolicy<host_exec_space>::type& policy,
    /// global tolerence parameters that governs all samples
    const real_type_1d_view_host& tol_newton,
    const real_type_2d_view_host& tol_time,
    /// sample specific input
    const real_type_2d_view_host& fac,
    const time_advance_type_1d_view_host& tadv,
    const real_type_2d_view_host& state,
    /// output
    const real_type_1d_view_host& t_out,
    const real_type_1d_view_host& dt_out,
    const real_type_2d_view_host& state_out,
    /// const data from kinetic model
    const Tines::value_type_1d_view<KineticModelConstData<interf_host_device_type>,interf_host_device_type>& kmcds);

  static void runHostBatchCVODE( /// input
    typename UseThisTeamPolicy<host_exec_space>::type& policy,
    /// global tolerence parameters that governs all samples
    const real_type_2d_view_host& tol_time,
    /// sample specific input
    const real_type_2d_view_host& fac,
    const time_advance_type_1d_view_host& tadv,
    const real_type_2d_view_host& state,
    /// output
    const real_type_1d_view_host& t_out,
    const real_type_1d_view_host& dt_out,
    const real_type_2d_view_host& state_out,
    /// const data from kinetic model
    const Tines::value_type_1d_view<KineticModelConstData<interf_host_device_type>,interf_host_device_type>& kmcds,
    const Tines::value_type_1d_view<Tines::TimeIntegratorCVODE<real_type,interf_host_device_type>,interf_host_device_type>& cvodes);

};

} // namespace TChem

#endif
