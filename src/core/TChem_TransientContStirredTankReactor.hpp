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

struct TransientContStirredTankReactorData
{
  real_type mdotIn; // inlet mass flow kg/s
  real_type Vol; // volumen of reactor m3
  real_type_1d_view Yi; // initial condition mass fraction
  real_type Acat; // Catalytic area m2: chemical active area
  real_type pressure;
  real_type EnthalpyIn;
  // real_type temperature;
};

using cstr_data_type = TransientContStirredTankReactorData;
using cstr_data_type_0d_dual_view =
  Kokkos::DualView<cstr_data_type, Kokkos::LayoutRight, exec_space>;
using cstr_data_type_1d_dual_view =
  Kokkos::DualView<cstr_data_type*, Kokkos::LayoutRight, exec_space>;
using cstr_data_type_0d_view = typename cstr_data_type_0d_dual_view::t_dev;
using cstr_data_type_0d_view_host = typename cstr_data_type_0d_dual_view::t_host;
using cstr_data_type_1d_view = typename cstr_data_type_1d_dual_view::t_dev;
using cstr_data_type_1d_view_host = typename cstr_data_type_1d_dual_view::t_host;

struct TransientContStirredTankReactor
{

  template<typename MemberType, typename RealType1DViewType>
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
           typename RealType1DViewType,
           typename RealType0DViewType>
  static KOKKOS_INLINE_FUNCTION void unpackFromValues(
    const MemberType& member,
    const RealType1DViewType& vals,
    const RealType0DViewType& temperature,
    /// input
    const RealType1DViewType& Ys,
    const RealType1DViewType& Zs)
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

  template<typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType,
           typename TransientContStirredTankReactorConstDataType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf,
    const TransientContStirredTankReactorConstDataType& cstr)
  {
    return (
      Impl::TransientContStirredTankReactor::getWorkSpaceSize(kmcd, kmcdSurf, cstr) +
      Impl::TransientContStirredTankReactor_Problem<
        KineticModelConstDataType,
        KineticSurfModelConstDataType,
        TransientContStirredTankReactorConstDataType>::getNumberOfEquations(kmcd, kmcdSurf));
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
    /// output
    const real_type_1d_view& t_out,
    const real_type_1d_view& dt_out,
    const real_type_2d_view& state_out,
    const real_type_2d_view& Z_out,
    /// const data from kinetic model
    const KineticModelConstDataDevice& kmcd,
    const KineticSurfModelConstDataDevice& kmcdSurf,
    const cstr_data_type& cstr);
};

} // namespace TChem

#endif
