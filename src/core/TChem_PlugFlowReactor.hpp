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
#ifndef __TCHEM_PLUGFLOWREACTOR_HPP__
#define __TCHEM_PLUGFLOWREACTOR_HPP__

#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"

#include "TChem_Impl_PlugFlowReactor.hpp"

namespace TChem {

struct PlugFlowReactorData
{
  real_type Area;
  real_type Pcat;
};
using pfr_data_type = PlugFlowReactorData;
using pfr_data_type_0d_dual_view =
  Kokkos::DualView<pfr_data_type, Kokkos::LayoutRight, exec_space>;
using pfr_data_type_0d_view = typename pfr_data_type_0d_dual_view::t_dev;
using pfr_data_type_0d_view_host = typename pfr_data_type_0d_dual_view::t_host;

// using pfr_data_type_1d_dual_view =
// Kokkos::DualView<pfr_data_type*,Kokkos::LayoutRight,exec_space>; using
// pfr_data_type_1d_view = typename pfr_data_type_1d_dual_view::t_dev; using
// pfr_data_type_1d_view_host = typename pfr_data_type_1d_dual_view::t_host;

struct PlugFlowReactor
{

  template<typename MemberType, typename RealType1DViewType>
  static KOKKOS_INLINE_FUNCTION void packToValues(
    const MemberType& member,
    /// input (pressure is not used)
    const real_type& temperature,
    const RealType1DViewType& Ys,
    const real_type& density,
    const real_type& velocity,
    const RealType1DViewType& Zs,
    /// output
    const RealType1DViewType& vals)
  {
    const ordinal_type m(Ys.extent(0) + 1);

    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                         [=](const ordinal_type& i) {
                           vals(i) = i == 0 ? temperature : Ys(i - 1);
                         });

    vals(m) = density;
    vals(m + 1) = velocity;
    const ordinal_type n(Zs.extent(0));

    Kokkos::parallel_for(
      Kokkos::TeamVectorRange(member, n),
      [=](const ordinal_type& i) { vals(i + m + 2) = Zs(i); });

    member.team_barrier();
  }

  template<typename MemberType,
           typename RealType0DViewType,
           typename RealType1DViewType>
  static KOKKOS_INLINE_FUNCTION void unpackFromValues(
    const MemberType& member,
    /// input
    const RealType1DViewType& vals,
    /// output (store back to state vector)
    const RealType0DViewType& temperature,
    const RealType1DViewType& Ys,
    const RealType0DViewType& density,
    const RealType0DViewType& velocity,
    const RealType1DViewType& Zs)
  {

    const ordinal_type m(Ys.extent(0) + 1);

    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                         [=](const ordinal_type& i) {
                           if (i == 0) {
                             temperature() = vals(0);
                           } else {
                             Ys(i - 1) = vals(i);
                           }
                         });
    density() = vals(m);
    velocity() = vals(m + 1);
    const ordinal_type n(Zs.extent(0));

    Kokkos::parallel_for(
      Kokkos::TeamVectorRange(member, n),
      [=](const ordinal_type& i) { Zs(i) = vals(i + m + 2); });
    member.team_barrier();
  }

  template<typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType,
           typename PlugFlowReactorConstDataType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf,
    const PlugFlowReactorConstDataType& pfrd)
  {
    return (
      Impl::PlugFlowReactor::getWorkSpaceSize(kmcd, kmcdSurf, pfrd) +
      Impl::PlugFlowReactor_Problem<
        KineticModelConstDataType,
        KineticSurfModelConstDataType,
        PlugFlowReactorConstDataType>::getNumberOfEquations(kmcd, kmcdSurf));
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
    const real_type_1d_view& tol_newton,
    const real_type_2d_view& tol_time,
    /// sample specific input
    const real_type_2d_view& fac,
    const time_advance_type_1d_view& tadv,
    const real_type_2d_view& state,
    const real_type_2d_view& zSurf,
    const real_type_1d_view& velocity,
    /// output
    const real_type_1d_view& t_out,
    const real_type_1d_view& dt_out,
    const real_type_2d_view& state_out,
    const real_type_2d_view& Z_out,
    const real_type_1d_view& velocity_out,
    /// const data from kinetic model
    const KineticModelConstDataDevice& kmcd,
    const KineticSurfModelConstDataDevice& kmcdSurf,
    // const pfr_data_type_0d_view& pfrd,
    const real_type Area,
    const real_type Pcat);

  //
  static void
  runDeviceBatch( /// thread block size
    typename UseThisTeamPolicy<exec_space>::type& policy,
    /// input
    const real_type_1d_view& tol_newton,
    const real_type_2d_view& tol_time,
    const real_type_2d_view& fac,
    const time_advance_type_1d_view& tadv,
    const real_type_2d_view& state,
    const real_type_2d_view& zSurf,
    const real_type_1d_view& velocity,
    /// output
    const real_type_1d_view& t_out,
    const real_type_1d_view& dt_out,
    const real_type_2d_view& state_out,
    const real_type_2d_view& Z_out,
    const real_type_1d_view& velocity_out,
    /// const data from kinetic model
    const Kokkos::View<KineticModelConstDataDevice*,exec_space>& kmcds,
    const Kokkos::View<KineticSurfModelConstDataDevice*,exec_space>& kmcdSurfs,
    // const pfr_data_type_0d_view& pfrd_info,
    const real_type Area,
    const real_type Pcat);
};

} // namespace TChem

#endif
