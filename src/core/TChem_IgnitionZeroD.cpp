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
#include "TChem_Util.hpp"
#include "TChem_IgnitionZeroD.hpp"

/// tadv - an input structure for time marching
/// state (nSpec+3) - initial condition of the state vector
/// qidx (lt nSpec+1) - QoI indices to store in qoi output
/// work - work space sized by getWorkSpaceSize
/// tcnt - time counter
/// qoi (time + qidx.extent(0)) - QoI output
/// kmcd - const data for kinetic model

namespace TChem {

  template<typename PolicyType,
           typename ValueType,
           typename DeviceType>
  void
  IgnitionZeroD_TemplateRunModelVariation( /// required template arguments
                                          const std::string& profile_name,
                                          const ValueType& dummyValueType,
                                          /// team size setting
                                          const PolicyType& policy,

                                          /// input
                                          const Tines::value_type_1d_view<real_type, DeviceType>& tol_newton,
                                          const Tines::value_type_2d_view<real_type, DeviceType>& tol_time,
                                          const Tines::value_type_2d_view<real_type, DeviceType>& fac,
                                          const Tines::value_type_1d_view<time_advance_type, DeviceType>& tadv,
                                          const Tines::value_type_2d_view<real_type, DeviceType>& state,
                                          /// output
                                          const Tines::value_type_1d_view<real_type, DeviceType>& t_out,
                                          const Tines::value_type_1d_view<real_type, DeviceType>& dt_out,
                                          const Tines::value_type_2d_view<real_type, DeviceType>& state_out,
                                          /// const data from kinetic model
                                          const Kokkos::View<KineticModelConstData<DeviceType >*,DeviceType>& kmcds);

#define TCHEM_RUN_IGNITION_ZERO_D_REACTOR_MODEL_VARIATION()     \
    IgnitionZeroD_TemplateRunModelVariation(                    \
                                            profile_name,       \
                                            value_type(),       \
                                            policy,             \
                                            tol_newton,         \
                                            tol_time,           \
                                            fac,                \
                                            tadv,               \
                                            state,              \
                                            t_out,              \
                                            dt_out,             \
                                            state_out,          \
                                            kmcds)


  template<typename PolicyType,
           typename ValueType,
           typename DeviceType>
  void
  IgnitionZeroD_TemplateRun( /// required template arguments
                            const std::string& profile_name,
                            const ValueType& dummyValueType,
                            /// team size setting
                            const PolicyType& policy,

                            /// input
                            const Tines::value_type_1d_view<real_type, DeviceType>& tol_newton,
                            const Tines::value_type_2d_view<real_type, DeviceType>& tol_time,
                            const Tines::value_type_2d_view<real_type, DeviceType>& fac,
                            const Tines::value_type_1d_view<time_advance_type, DeviceType>& tadv,
                            const Tines::value_type_2d_view<real_type, DeviceType>& state,
                            /// output
                            const Tines::value_type_1d_view<real_type, DeviceType>& t_out,
                            const Tines::value_type_1d_view<real_type, DeviceType>& dt_out,
                            const Tines::value_type_2d_view<real_type, DeviceType>& state_out,
                            /// const data from kinetic model
                            const KineticModelConstData<DeviceType>& kmcd);

#define TCHEM_RUN_IGNITION_ZERO_D_REACTOR()     \
    IgnitionZeroD_TemplateRun(                  \
                              profile_name,     \
                              value_type(),     \
                              policy,           \
                              tol_newton,       \
                              tol_time,         \
                              fac,              \
                              tadv,             \
                              state,            \
                              t_out,            \
                              dt_out,           \
                              state_out,        \
                              kmcd)

  void
  IgnitionZeroD::runDeviceBatch( /// thread block size
                                typename UseThisTeamPolicy<exec_space>::type& policy,
                                /// input
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
                                const KineticModelConstData<interf_device_type >& kmcd)
  {
    const std::string profile_name = "TChem::IgnitionZeroD::runDeviceBatch::kmcd";

#if defined(TCHEM_ENABLE_SACADO_JACOBIAN_IGNITION_ZERO_D_REACTOR)
    using problem_type = Impl::IgnitionZeroD_Problem<real_type, interf_device_type>;
    const ordinal_type m = problem_type::getNumberOfEquations(kmcd);

    if (m < 16) {
      using value_type = Sacado::Fad::SLFad<real_type,16>;
      TCHEM_RUN_IGNITION_ZERO_D_REACTOR();
    } else if  (m < 32) {
      using value_type = Sacado::Fad::SLFad<real_type,32>;
      TCHEM_RUN_IGNITION_ZERO_D_REACTOR();
    } else if  (m < 64) {
      using value_type = Sacado::Fad::SLFad<real_type,64>;
      TCHEM_RUN_IGNITION_ZERO_D_REACTOR();
    } else if  (m < 128) {
      using value_type = Sacado::Fad::SLFad<real_type,128>;
      TCHEM_RUN_IGNITION_ZERO_D_REACTOR();
    } else if  (m < 256) {
      using value_type = Sacado::Fad::SLFad<real_type,256>;
      TCHEM_RUN_IGNITION_ZERO_D_REACTOR();
    } else if  (m < 512) {
      using value_type = Sacado::Fad::SLFad<real_type,512>;
      TCHEM_RUN_IGNITION_ZERO_D_REACTOR();
    } else if (m < 1024){
      using value_type = Sacado::Fad::SLFad<real_type,1024>;
      TCHEM_RUN_IGNITION_ZERO_D_REACTOR();
    } else{
      TCHEM_CHECK_ERROR(0,
                        "Error: Number of equations is bigger than size of sacado fad type");
    }
#else
    using value_type = real_type;
    TCHEM_RUN_IGNITION_ZERO_D_REACTOR();
#endif
  }

  void
  IgnitionZeroD::runHostBatch( /// input
                              typename UseThisTeamPolicy<host_exec_space>::type& policy,
                              const real_type_1d_view_host& tol_newton,
                              const real_type_2d_view_host& tol_time,
                              const real_type_2d_view_host& fac,
                              const time_advance_type_1d_view_host& tadv,
                              const real_type_2d_view_host& state,
                              /// output
                              const real_type_1d_view_host& t_out,
                              const real_type_1d_view_host& dt_out,
                              const real_type_2d_view_host& state_out,
                              /// const data from kinetic model
                              const KineticModelConstData<interf_host_device_type>& kmcd)
  {
    const std::string profile_name = "TChem::IgnitionZeroD::runHostBatch::kmcd";
#if defined(TCHEM_ENABLE_SACADO_JACOBIAN_IGNITION_ZERO_D_REACTOR)
    using problem_type = Impl::IgnitionZeroD_Problem<real_type, interf_host_device_type>;
    const ordinal_type m = problem_type::getNumberOfEquations(kmcd);

    if (m < 16) {
      using value_type = Sacado::Fad::SLFad<real_type,16>;
      TCHEM_RUN_IGNITION_ZERO_D_REACTOR();
    } else if  (m < 32) {
      using value_type = Sacado::Fad::SLFad<real_type,32>;
      TCHEM_RUN_IGNITION_ZERO_D_REACTOR();
    } else if  (m < 64) {
      using value_type = Sacado::Fad::SLFad<real_type,64>;
      TCHEM_RUN_IGNITION_ZERO_D_REACTOR();
    } else if  (m < 128) {
      using value_type = Sacado::Fad::SLFad<real_type,128>;
      TCHEM_RUN_IGNITION_ZERO_D_REACTOR();
    } else if  (m < 256) {
      using value_type = Sacado::Fad::SLFad<real_type,256>;
      TCHEM_RUN_IGNITION_ZERO_D_REACTOR();
    } else if  (m < 512) {
      using value_type = Sacado::Fad::SLFad<real_type,512>;
      TCHEM_RUN_IGNITION_ZERO_D_REACTOR();
    } else if (m < 1024){
      using value_type = Sacado::Fad::SLFad<real_type,1024>;
      TCHEM_RUN_IGNITION_ZERO_D_REACTOR();
    } else{
      TCHEM_CHECK_ERROR(0,
                        "Error: Number of equations is bigger than size of sacado fad type");
    }
#else
    using value_type = real_type;
    TCHEM_RUN_IGNITION_ZERO_D_REACTOR();
#endif
  }

  void
  IgnitionZeroD::runDeviceBatch( /// thread block size
                                typename UseThisTeamPolicy<exec_space>::type& policy,
                                /// input
                                const real_type_1d_view& tol_newton,
                                const real_type_2d_view& tol_time,
                                const real_type_2d_view& fac,
                                const time_advance_type_1d_view& tadv,
                                const real_type_2d_view& state,
                                /// output
                                const real_type_1d_view& t_out,
                                const real_type_1d_view& dt_out,
                                const real_type_2d_view& state_out,
                                /// const data from kinetic model
                                const Kokkos::View<KineticModelConstData<interf_device_type>*,interf_device_type>& kmcds)
  {
    const std::string profile_name = "TChem::IgnitionZeroD::runDeviceBatch::kmcd array";

#if defined(TCHEM_ENABLE_SACADO_JACOBIAN_IGNITION_ZERO_D_REACTOR)
    using problem_type = Impl::IgnitionZeroD_Problem<real_type, interf_device_type>;
    const ordinal_type m = problem_type::getNumberOfEquations(kmcds(0));

    if (m < 128) {
      using value_type = Sacado::Fad::SLFad<real_type,128>;
      TCHEM_RUN_IGNITION_ZERO_D_REACTOR_MODEL_VARIATION();
    } else if  (m < 256) {
      using value_type = Sacado::Fad::SLFad<real_type,256>;
      TCHEM_RUN_IGNITION_ZERO_D_REACTOR_MODEL_VARIATION();
    } else if  (m < 512) {
      using value_type = Sacado::Fad::SLFad<real_type,512>;
      TCHEM_RUN_IGNITION_ZERO_D_REACTOR_MODEL_VARIATION();
    } else if (m < 1024){
      using value_type = Sacado::Fad::SLFad<real_type,1024>;
      TCHEM_RUN_IGNITION_ZERO_D_REACTOR_MODEL_VARIATION();
    } else{
      TCHEM_CHECK_ERROR(0,
                        "Error: Number of equations is bigger than size of sacado fad type");
    }
#else
    using value_type = real_type;
    TCHEM_RUN_IGNITION_ZERO_D_REACTOR_MODEL_VARIATION();
#endif

  }

  void
  IgnitionZeroD::runHostBatch( /// thread block size
                              typename UseThisTeamPolicy<host_exec_space>::type& policy,
                              /// input
                              const real_type_1d_view_host& tol_newton,
                              const real_type_2d_view_host& tol_time,
                              const real_type_2d_view_host& fac,
                              const time_advance_type_1d_view_host& tadv,
                              const real_type_2d_view_host& state,
                              /// output
                              const real_type_1d_view_host& t_out,
                              const real_type_1d_view_host& dt_out,
                              const real_type_2d_view_host& state_out,
                              /// const data from kinetic model
                              const Kokkos::View<KineticModelConstData<interf_host_device_type>*,interf_host_device_type>& kmcds)
  {
    const std::string profile_name = "TChem::IgnitionZeroD::runHostBatch::kmcd array";

#if defined(TCHEM_ENABLE_SACADO_JACOBIAN_IGNITION_ZERO_D_REACTOR)
    using problem_type = Impl::IgnitionZeroD_Problem<real_type, interf_host_device_type>;
    const ordinal_type m = problem_type::getNumberOfEquations(kmcds(0));

    if (m < 128) {
      using value_type = Sacado::Fad::SLFad<real_type,128>;
      TCHEM_RUN_IGNITION_ZERO_D_REACTOR_MODEL_VARIATION();
    } else if  (m < 256) {
      using value_type = Sacado::Fad::SLFad<real_type,256>;
      TCHEM_RUN_IGNITION_ZERO_D_REACTOR_MODEL_VARIATION();
    } else if  (m < 512) {
      using value_type = Sacado::Fad::SLFad<real_type,512>;
      TCHEM_RUN_IGNITION_ZERO_D_REACTOR_MODEL_VARIATION();
    } else if (m < 1024){
      using value_type = Sacado::Fad::SLFad<real_type,1024>;
      TCHEM_RUN_IGNITION_ZERO_D_REACTOR_MODEL_VARIATION();
    } else{
      TCHEM_CHECK_ERROR(0,
                        "Error: Number of equations is bigger than size of sacado fad type");
    }
#else
    using value_type = real_type;
    TCHEM_RUN_IGNITION_ZERO_D_REACTOR_MODEL_VARIATION();
#endif

  }

} // namespace TChem
