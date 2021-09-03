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
#ifndef __TCHEM_TRANSIENT_CONT_STIRRED_TANK_REACTOR_HPP__
#define __TCHEM_TRANSIENT_CONT_STIRRED_TANK_REACTOR_HPP__

#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"

#include "TChem_Impl_TransientContStirredTankReactor.hpp"

namespace TChem {

struct TransientContStirredTankReactor
{

  using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;
  using device_type      = typename Tines::UseThisDevice<exec_space>::type;
  using real_type_0d_view_type = Tines::value_type_0d_view<real_type,device_type>;
  using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;
  using real_type_2d_view_type = Tines::value_type_2d_view<real_type,device_type>;

  using kinetic_model_type = KineticModelConstData<device_type>;
  using kinetic_surf_model_type = KineticSurfModelConstData<device_type>;

  template<typename MemberType,
           typename RealType1DViewType>
  static KOKKOS_INLINE_FUNCTION void packToValues(
    const MemberType& member,
    const real_type& temperature,
    const RealType1DViewType& Ys,
    const RealType1DViewType& Zs,
    /// output
    const RealType1DViewType& vals)
  {
    vals(0) = temperature;

    const ordinal_type m(Ys.extent(0));

    Kokkos::parallel_for(
      Kokkos::TeamVectorRange(member, m),
      [&](const ordinal_type& i) {vals(i + 1) = Ys(i);});

    const ordinal_type n(Zs.extent(0));

    Kokkos::parallel_for(
      Kokkos::TeamVectorRange(member, n),
      [=](const ordinal_type& i) { vals(i + m + 1 ) = Zs(i); });

    member.team_barrier();
  }

  template<typename MemberType,
           typename DeviceType>
  static KOKKOS_INLINE_FUNCTION void unpackFromValues(
    const MemberType& member,
    const Tines::value_type_1d_view<real_type,DeviceType>& vals,
    const Tines::value_type_0d_view<real_type,DeviceType>& temperature,
    /// input
    const Tines::value_type_1d_view<real_type,DeviceType>& Ys,
    const Tines::value_type_1d_view<real_type,DeviceType>& Zs)
  {

    temperature() = vals(0);
    const ordinal_type m(Ys.extent(0));

    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                         [&](const ordinal_type& i) {
                             Ys(i) = vals(i + 1);
                         });
    const ordinal_type n(Zs.extent(0));

    Kokkos::parallel_for(
      Kokkos::TeamVectorRange(member, n),
      [=](const ordinal_type& i) { Zs(i) = vals(i + m + 1); });
    member.team_barrier();
  }

  template< typename DeviceType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstData<DeviceType>& kmcd,
    const KineticSurfModelConstData<DeviceType>& kmcdSurf)
  {
    using device_type = DeviceType;
    const ordinal_type m = Impl::TransientContStirredTankReactor_Problem
      <real_type,device_type>::getNumberOfEquations(kmcd, kmcdSurf);
    ordinal_type work_size(0) ;
//
#if defined(TCHEM_ENABLE_SACADO_JACOBIAN_TRANSIENT_CONT_STIRRED_TANK_REACTOR)
    if (m < 128) {
      using value_type = Sacado::Fad::SLFad<real_type,128>;
      work_size = Impl::TransientContStirredTankReactor
        <value_type, device_type>::getWorkSpaceSize(kmcd, kmcdSurf)  + m ;
    } else if  (m < 256) {
      using value_type = Sacado::Fad::SLFad<real_type,256>;
      work_size = Impl::TransientContStirredTankReactor
        <value_type, device_type>::getWorkSpaceSize(kmcd, kmcdSurf)  + m ;
    } else if  (m < 512) {
      using value_type = Sacado::Fad::SLFad<real_type,512>;
      work_size = Impl::TransientContStirredTankReactor
        <value_type, device_type>::getWorkSpaceSize(kmcd, kmcdSurf)  + m ;
    } else if (m < 1024){
      using value_type = Sacado::Fad::SLFad<real_type,1024>;
      work_size = Impl::TransientContStirredTankReactor
        <value_type, device_type>::getWorkSpaceSize(kmcd, kmcdSurf)  + m ;
    } else{
      TCHEM_CHECK_ERROR(0,
                        "Error: Number of equations is bigger than size of sacado fad type");
    }
#else
    {
    work_size =  Impl::TransientContStirredTankReactor
      <real_type,device_type>::getWorkSpaceSize(kmcd, kmcdSurf) + m;
    }
#endif
    return work_size ;
  }

  /// tadv - an input structure for time marching
  /// state (nSpec+3) - initial condition of the state vector
  /// work - work space sized by getWorkSpaceSize
  /// t_out - time when this code exits
  /// state_out - final condition of the state vector (the same input state can
  /// be overwritten) kmcd - const data for kinetic model

  static void runDeviceBatch( /// thread block size
    typename UseThisTeamPolicy<exec_space>::type& policy,
    /// input
    const real_type_1d_view_type& tol_newton,
    const real_type_2d_view_type& tol_time,
    /// sample specific input
    const real_type_2d_view_type& fac,
    const time_advance_type_1d_view& tadv,
    const real_type_2d_view_type& state,
    const real_type_2d_view_type& site_fraction,
    /// output
    const real_type_1d_view_type& t_out,
    const real_type_1d_view_type& dt_out,
    const real_type_2d_view_type& state_out,
    const real_type_2d_view_type& site_fraction_out,
    /// const data from kinetic model
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf,
    const TransientContStirredTankReactorData<device_type>& cstr);

  //
  static void
  runDeviceBatch( /// thread block size
    typename UseThisTeamPolicy<exec_space>::type& policy,
    /// input
    const real_type_1d_view_type& tol_newton,
    const real_type_2d_view_type& tol_time,
    const real_type_2d_view_type& fac,
    const time_advance_type_1d_view& tadv,
    const real_type_2d_view_type& state,
    const real_type_2d_view_type& site_fraction,
    /// output
    const real_type_1d_view_type& t_out,
    const real_type_1d_view_type& dt_out,
    const real_type_2d_view_type& state_out,
    const real_type_2d_view_type& site_fraction_out,
    /// const data from kinetic model
    const Kokkos::View<kinetic_model_type*,device_type>& kmcds,
    const Kokkos::View<kinetic_surf_model_type*,device_type>& kmcdSurfs,
    const TransientContStirredTankReactorData<device_type>& cstr);
};


} // namespace TChem

#endif
