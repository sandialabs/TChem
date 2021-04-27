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

#include "TChem_PlugFlowReactor.hpp"
#include "TChem_PlugFlowReactorRHS.hpp"

namespace TChem {

template<typename PolicyType,
         typename TimeAdvance1DViewType,
         typename RealType0DViewType,
         typename RealType1DViewType,
         typename RealType2DViewType,
         typename KineticModelConstType,
         typename KineticSurfModelConstData,
         typename PlugFlowReactorConstDataType>
void
PlugFlowReactor_TemplateRun( /// required template arguments
  const std::string& profile_name,
  const RealType0DViewType& dummy_0d,
  /// team size setting
  const PolicyType& policy,
  // inputs
  const RealType1DViewType& tol_newton,
  const RealType2DViewType& tol_time,
  const RealType2DViewType& fac,
  const TimeAdvance1DViewType& tadv,
  const RealType2DViewType& state,
  const RealType2DViewType& zSurf,
  const RealType1DViewType& velocity,
  /// output
  const RealType1DViewType& t_out,
  const RealType1DViewType& dt_out,
  const RealType2DViewType& state_out,
  const RealType2DViewType& Z_out,
  const RealType1DViewType& velocity_out,
  /// const data from kinetic model
  const KineticModelConstType& kmcd,
  const KineticSurfModelConstData& kmcdSurf,
  const PlugFlowReactorConstDataType& pfrd)
{
  Kokkos::Profiling::pushRegion(profile_name);
  using policy_type = PolicyType;

  const ordinal_type level = 1;

  const ordinal_type m = Impl::PlugFlowReactor_Problem<
    KineticModelConstType,
    KineticSurfModelConstData,
    PlugFlowReactorConstDataType>::getNumberOfEquations(kmcd, kmcdSurf);
  // const ordinal_type n = Impl::PlugFlowReactor_Problem<
  //   KineticModelConstType,
  //   KineticSurfModelConstData,
  //   PlugFlowReactorConstDataType>::getNumberOfTimeODEs(kmcd);

  const ordinal_type per_team_extent = TChem::PlugFlowReactor::getWorkSpaceSize(
    kmcd,
    kmcdSurf,
    pfrd); /// this +m seems to be included in the workspace size calculation
  // const ordinal_type per_team_scratch =
  //   Scratch<RealType1DViewType>::shmem_size(per_team_extent);

  // policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());
  // if (team_size > 0 && vector_size > 0) {
  //   policy = policy_type(exec_space_instance, nBatch, team_size,
  //   vector_size);
  // }
  // policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

  Kokkos::parallel_for(
    profile_name,
    policy,
    KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
      const ordinal_type i = member.league_rank();
      const RealType1DViewType fac_at_i =
        Kokkos::subview(fac, i, Kokkos::ALL());
      const auto tadv_at_i = tadv(i);
      const real_type /*t_beg = tadv_at_i._tbeg, */t_end = tadv_at_i._tend;
      const RealType0DViewType t_out_at_i = Kokkos::subview(t_out, i);
      if (t_out_at_i() < t_end) {
      const RealType1DViewType state_at_i =
        Kokkos::subview(state, i, Kokkos::ALL());
      const RealType0DViewType dt_out_at_i = Kokkos::subview(dt_out, i);

      // site fraction
      const RealType1DViewType Zs_at_i =
        Kokkos::subview(zSurf, i, Kokkos::ALL());

      Scratch<RealType1DViewType> work(member.team_scratch(level),
                                       per_team_extent);

      Impl::StateVector<RealType1DViewType> sv_at_i(kmcd.nSpec, state_at_i);
      TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                        "Error: input state vector is not valid");
      {
        const ordinal_type max_num_newton_iterations =
          tadv_at_i._max_num_newton_iterations;
        const ordinal_type max_num_time_iterations =
          tadv_at_i._num_time_iterations_per_interval;

        const real_type dt_in = tadv_at_i._dt, dt_min = tadv_at_i._dtmin,
                        dt_max = tadv_at_i._dtmax;
	const real_type t_beg = tadv_at_i._tbeg;

        const real_type temperature = sv_at_i.Temperature();
        const real_type pressure = sv_at_i.Pressure();
        const real_type density = sv_at_i.Density();
        const RealType1DViewType Ys = sv_at_i.MassFractions();
        const real_type vel_at_i = velocity(i);

        // printf("velocity at batch run % e\n", vel_at_i );

        const RealType0DViewType temperature_out(sv_at_i.TemperaturePtr());
        const RealType0DViewType pressure_out(sv_at_i.PressurePtr());
        const RealType0DViewType density_out(sv_at_i.DensityPtr());
        const RealType1DViewType Ys_out = Ys;

        const RealType1DViewType Zs_out_at_i = Zs_at_i;
        const RealType0DViewType vel_out_at_i =
          Kokkos::subview(velocity_out, i);

        auto wptr = work.data();
        const RealType1DViewType vals(wptr, m);
        wptr += m;
        const RealType1DViewType ww(wptr,
                                    work.extent(0) - (wptr - work.data()));
	if (t_out_at_i() < t_end) {
	  TChem::PlugFlowReactor::packToValues(
            member, temperature, Ys, density, vel_at_i, Zs_at_i, vals);

        member.team_barrier();
        TChem::Impl::PlugFlowReactor ::team_invoke(member,
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
                                                   vals,
                                                   t_out_at_i,
                                                   dt_out_at_i,
                                                   vals,
                                                   ww, // work
                                                   kmcd,
                                                   kmcdSurf,
                                                   pfrd);

        member.team_barrier();
        TChem::PlugFlowReactor::unpackFromValues(member,
                                                 vals,
                                                 temperature_out,
                                                 Ys_out,
                                                 density_out,
                                                 vel_out_at_i,
                                                 Zs_at_i);
        // pressure is a depend variable. Thus, it is not part of vars
        // Compute pressure for futher analysis (csp)
        member.team_barrier();
        const real_type Wmix =
          Impl::MolarWeights ::team_invoke(member, Ys_out, kmcd);
        pressure_out() = kmcd.Runiv * temperature_out() * density_out() /
                         Wmix; // compute pressure
	}
      }
      }
    });
  Kokkos::Profiling::popRegion();
}

template<typename PolicyType,
         typename TimeAdvance1DViewType,
         typename RealType0DViewType,
         typename RealType1DViewType,
         typename RealType2DViewType,
         typename KineticModelConstViewType,
         typename KineticSurfModelConstViewType,
         typename PlugFlowReactorConstDataType>
void
PlugFlowReactor_TemplateRunModelVariation( /// required template arguments
  const std::string& profile_name,
  const RealType0DViewType& dummy_0d,
  /// team size setting
  const PolicyType& policy,
  // inputs
  const RealType1DViewType& tol_newton,
  const RealType2DViewType& tol_time,
  const RealType2DViewType& fac,
  const TimeAdvance1DViewType& tadv,
  const RealType2DViewType& state,
  const RealType2DViewType& zSurf,
  const RealType1DViewType& velocity,
  /// output
  const RealType1DViewType& t_out,
  const RealType1DViewType& dt_out,
  const RealType2DViewType& state_out,
  const RealType2DViewType& Z_out,
  const RealType1DViewType& velocity_out,
  /// const data from kinetic model
  const KineticModelConstViewType& kmcds,
  const KineticSurfModelConstViewType& kmcdSurfs,
  const PlugFlowReactorConstDataType& pfrd)
{
  Kokkos::Profiling::pushRegion(profile_name);
  using policy_type = PolicyType;

  const ordinal_type level = 1;

  const ordinal_type m = kmcds(0).nSpec + 3 + kmcdSurfs(0).nSpec;

  const ordinal_type per_team_extent = TChem::PlugFlowReactor::getWorkSpaceSize(
    kmcds(0),
    kmcdSurfs(0),
    pfrd);

  Kokkos::parallel_for(
    profile_name,
    policy,
    KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
      const ordinal_type i = member.league_rank();

      const auto kmcd_at_i = (kmcds.extent(0) == 1 ? kmcds(0) : kmcds(i));
      const auto kmcd_surf_at_i = (kmcdSurfs.extent(0) == 1 ? kmcdSurfs(0) : kmcdSurfs(i));

      const RealType1DViewType fac_at_i =
        Kokkos::subview(fac, i, Kokkos::ALL());
      const auto tadv_at_i = tadv(i);
      const real_type /*t_beg = tadv_at_i._tbeg, */t_end = tadv_at_i._tend;
      const RealType0DViewType t_out_at_i = Kokkos::subview(t_out, i);
      if (t_out_at_i() < t_end) {
      const RealType1DViewType state_at_i =
        Kokkos::subview(state, i, Kokkos::ALL());
      const RealType0DViewType dt_out_at_i = Kokkos::subview(dt_out, i);

      // site fraction
      const RealType1DViewType Zs_at_i =
        Kokkos::subview(zSurf, i, Kokkos::ALL());

      Scratch<RealType1DViewType> work(member.team_scratch(level),
                                       per_team_extent);

      Impl::StateVector<RealType1DViewType> sv_at_i(kmcd_at_i.nSpec, state_at_i);
      TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                        "Error: input state vector is not valid");
      {
        const ordinal_type max_num_newton_iterations =
          tadv_at_i._max_num_newton_iterations;
        const ordinal_type max_num_time_iterations =
          tadv_at_i._num_time_iterations_per_interval;

        const real_type dt_in = tadv_at_i._dt, dt_min = tadv_at_i._dtmin,
                        dt_max = tadv_at_i._dtmax;
	const real_type t_beg = tadv_at_i._tbeg;

        const real_type temperature = sv_at_i.Temperature();
        const real_type pressure = sv_at_i.Pressure();
        const real_type density = sv_at_i.Density();
        const RealType1DViewType Ys = sv_at_i.MassFractions();
        const real_type vel_at_i = velocity(i);

        // printf("velocity at batch run % e\n", vel_at_i );

        const RealType0DViewType temperature_out(sv_at_i.TemperaturePtr());
        const RealType0DViewType pressure_out(sv_at_i.PressurePtr());
        const RealType0DViewType density_out(sv_at_i.DensityPtr());
        const RealType1DViewType Ys_out = Ys;

        const RealType1DViewType Zs_out_at_i = Zs_at_i;
        const RealType0DViewType vel_out_at_i =
          Kokkos::subview(velocity_out, i);

        auto wptr = work.data();
        const RealType1DViewType vals(wptr, m);
        wptr += m;
        const RealType1DViewType ww(wptr,
                                    work.extent(0) - (wptr - work.data()));
	if (t_out_at_i() < t_end) {
	  TChem::PlugFlowReactor::packToValues(
            member, temperature, Ys, density, vel_at_i, Zs_at_i, vals);

        member.team_barrier();
        TChem::Impl::PlugFlowReactor ::team_invoke(member,
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
                                                   vals,
                                                   t_out_at_i,
                                                   dt_out_at_i,
                                                   vals,
                                                   ww, // work
                                                   kmcd_at_i,
                                                   kmcd_surf_at_i,
                                                   pfrd);

        member.team_barrier();
        TChem::PlugFlowReactor::unpackFromValues(member,
                                                 vals,
                                                 temperature_out,
                                                 Ys_out,
                                                 density_out,
                                                 vel_out_at_i,
                                                 Zs_at_i);
        // pressure is a depend variable. Thus, it is not part of vars
        // Compute pressure for futher analysis (csp)
        member.team_barrier();
        const real_type Wmix =
          Impl::MolarWeights ::team_invoke(member, Ys_out, kmcd_at_i);
        pressure_out() = kmcd_at_i.Runiv * temperature_out() * density_out() /
                         Wmix; // compute pressure
	}
      }
      }
    });
  Kokkos::Profiling::popRegion();
}


void
PlugFlowReactor::runDeviceBatch( /// thread block size
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
  const KineticModelConstDataDevice& kmcd,
  const KineticSurfModelConstDataDevice& kmcdSurf,
  // const pfr_data_type_0d_view& pfrd_info,
  const real_type Area,
  const real_type Pcat)
{

  // data for PRF

  // const pfr_data_type pfrd = pfrd_info();
  pfr_data_type pfrd;
  pfrd.Area = Area;
  pfrd.Pcat = Pcat;

  PlugFlowReactor_TemplateRun( /// template arguments deduction
    "TChem::PlugFlowReactor::runDeviceBatch",
    real_type_0d_view(),
    /// team policy
    policy,
    /// input
    tol_newton,
    tol_time,
    fac,
    tadv,
    state,
    zSurf,
    velocity,
    /// output
    t_out,
    dt_out,
    state_out,
    Z_out,
    velocity_out,
    /// const data of kinetic model
    kmcd,
    kmcdSurf,
    pfrd);
}

void
PlugFlowReactor::runDeviceBatch( /// thread block size
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
  const real_type Pcat)
{

  // data for PRF

  // const pfr_data_type pfrd = pfrd_info();
  pfr_data_type pfrd;
  pfrd.Area = Area;
  pfrd.Pcat = Pcat;

  PlugFlowReactor_TemplateRunModelVariation( /// template arguments deduction
    "TChem::PlugFlowReactor::runDeviceBatch array",
    real_type_0d_view(),
    /// team policy
    policy,
    /// input
    tol_newton,
    tol_time,
    fac,
    tadv,
    state,
    zSurf,
    velocity,
    /// output
    t_out,
    dt_out,
    state_out,
    Z_out,
    velocity_out,
    /// const data of kinetic model
    kmcds,
    kmcdSurfs,
    pfrd);
}

} // namespace TChem
