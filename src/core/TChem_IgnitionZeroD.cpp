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
         typename TimeAdvance1DViewType,
         typename RealType0DViewType,
         typename RealType1DViewType,
         typename RealType2DViewType,
         typename KineticModelConstType>
void
IgnitionZeroD_TemplateRun( /// required template arguments
  const std::string& profile_name,
  const RealType0DViewType& dummy_0d,
  /// team size setting
  const PolicyType& policy,
  /// input
  const RealType1DViewType& tol_newton,
  const RealType2DViewType& tol_time,
  const RealType2DViewType& fac,
  const TimeAdvance1DViewType& tadv,
  const RealType2DViewType& state,
  /// output
  const RealType1DViewType& t_out,
  const RealType1DViewType& dt_out,
  const RealType2DViewType& state_out,
  /// const data from kinetic model
  const KineticModelConstType& kmcd)
{
  Kokkos::Profiling::pushRegion(profile_name);
  using policy_type = PolicyType;

  const ordinal_type level = 1;
  const ordinal_type per_team_extent = IgnitionZeroD::getWorkSpaceSize(kmcd);

  Kokkos::parallel_for(
    profile_name,
    policy,
    KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
      const ordinal_type i = member.league_rank();
      const RealType1DViewType fac_at_i =
        Kokkos::subview(fac, i, Kokkos::ALL());
      const auto tadv_at_i = tadv(i);
      const real_type t_end = tadv_at_i._tend;
      const RealType0DViewType t_out_at_i = Kokkos::subview(t_out, i);
      if (t_out_at_i() < t_end) {
      const RealType1DViewType state_at_i =
        Kokkos::subview(state, i, Kokkos::ALL());
      const RealType1DViewType state_out_at_i =
        Kokkos::subview(state_out, i, Kokkos::ALL());

      const RealType0DViewType dt_out_at_i = Kokkos::subview(dt_out, i);
      Scratch<RealType1DViewType> work(member.team_scratch(level),
                                       per_team_extent);

      Impl::StateVector<RealType1DViewType> sv_at_i(kmcd.nSpec, state_at_i);
      Impl::StateVector<RealType1DViewType> sv_out_at_i(kmcd.nSpec,
                                                        state_out_at_i);
      TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                        "Error: input state vector is not valid");
      TCHEM_CHECK_ERROR(!sv_out_at_i.isValid(),
                        "Error: input state vector is not valid");
      {
        const ordinal_type max_num_newton_iterations =
          tadv_at_i._max_num_newton_iterations;
        const ordinal_type max_num_time_iterations =
          tadv_at_i._num_time_iterations_per_interval;

        const real_type dt_in = tadv_at_i._dt, dt_min = tadv_at_i._dtmin,
                        dt_max = tadv_at_i._dtmax;
	const real_type t_beg = tadv_at_i._tbeg;
	
        const auto temperature = sv_at_i.Temperature();
        const auto pressure = sv_at_i.Pressure();
        const auto Ys = sv_at_i.MassFractions();

        const RealType0DViewType temperature_out(sv_out_at_i.TemperaturePtr());
        const RealType0DViewType pressure_out(sv_out_at_i.PressurePtr());
        const RealType1DViewType Ys_out = sv_out_at_i.MassFractions();

        const ordinal_type m = Impl::IgnitionZeroD_Problem<
          KineticModelConstType>::getNumberOfEquations(kmcd);
        auto wptr = work.data();
        const RealType1DViewType vals(wptr, m);
        wptr += m;
        const RealType1DViewType ww(wptr,
                                    work.extent(0) - (wptr - work.data()));

        /// we can only guarantee vals is contiguous array. we basically assume
        /// that a state vector can be arbitrary ordered.

        /// m is nSpec + 1
        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                             [&](const ordinal_type& i) {
                               vals(i) = i == 0 ? temperature : Ys(i - 1);
                             });
        member.team_barrier();

        Impl::IgnitionZeroD ::team_invoke(member,
                                          max_num_newton_iterations,
                                          max_num_time_iterations,
                                          tol_newton,
                                          tol_time,
                                          fac_at_i,
                                          dt_in,
                                          dt_min,
                                          dt_max,
                                          t_beg,
                                          t_end,
                                          pressure,
                                          vals,
                                          t_out_at_i,
                                          dt_out_at_i,
                                          pressure_out,
                                          vals,
                                          ww,
                                          kmcd);

        member.team_barrier();
        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                             [&](const ordinal_type& i) {
                               if (i == 0) {
                                 temperature_out() = vals(0);
                               } else {
                                 Ys_out(i - 1) = vals(i);
                               }
                             });
        member.team_barrier();
      }
      }
    });
  Kokkos::Profiling::popRegion();
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
  const KineticModelConstDataHost& kmcd)
{
  IgnitionZeroD_TemplateRun( /// template arguments deduction
    "TChem::IgnitionZeroD::runHostBatch",
    real_type_0d_view_host(),
    /// team policy
    policy,
    /// input
    tol_newton,
    tol_time,
    fac,
    tadv,
    state,
    /// output
    t_out,
    dt_out,
    state_out,
    /// const data of kinetic model
    kmcd);
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
  const KineticModelConstDataDevice& kmcd)
{
  IgnitionZeroD_TemplateRun( /// template arguments deduction
    "TChem::IgnitionZeroD::runHostBatch",
    real_type_0d_view(),
    /// team policy
    policy,
    /// input
    tol_newton,
    tol_time,
    fac,
    tadv,
    state,
    /// output
    t_out,
    dt_out,
    state_out,
    /// const data of kinetic model
    kmcd);
}

} // namespace TChem
