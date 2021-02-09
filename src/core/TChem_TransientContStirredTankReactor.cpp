/* =====================================================================================
TChem version 2.1.0
Copyright (2020) NTESS
https://github.com/sandialabs/TChem

Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains 
certain rights in this software.

This file is part of TChem. TChem is open-source software: you can redistribute it
and/or modify it under the terms of BSD 2-Clause License
(https://opensource.org/licenses/BSD-2-Clause). A copy of the license is also
provided under the main directory

Questions? Contact Cosmin Safta at <csafta@sandia.gov>, or
           Kyungjoo Kim at <kyukim@sandia.gov>, or
           Oscar Diaz-Ibarra at <odiazib@sandia.gov>

Sandia National Laboratories, Livermore, CA, USA
===================================================================================== */


#include "TChem_Util.hpp"

#include "TChem_TransientContStirredTankReactor.hpp"
#include "TChem_Impl_TransientContStirredTankReactorRHS.hpp"

namespace TChem {

template<typename PolicyType,
         typename TimeAdvance1DViewType,
         typename RealType0DViewType,
         typename RealType1DViewType,
         typename RealType2DViewType,
         typename KineticModelConstType,
         typename KineticSurfModelConstData,
         typename TransientContStirredTankReactorConstDataType>
void
TransientContStirredTankReactor_TemplateRun( /// required template arguments
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
  /// output
  const RealType1DViewType& t_out,
  const RealType1DViewType& dt_out,
  const RealType2DViewType& state_out,
  const RealType2DViewType& Z_out,
  /// const data from kinetic model
  const KineticModelConstType& kmcd,
  const KineticSurfModelConstData& kmcdSurf,
  const TransientContStirredTankReactorConstDataType& cstr)
{
  Kokkos::Profiling::pushRegion(profile_name);
  using policy_type = PolicyType;

  const ordinal_type level = 1;

  const ordinal_type m = Impl::TransientContStirredTankReactor_Problem<
    KineticModelConstType,
    KineticSurfModelConstData,
    TransientContStirredTankReactorConstDataType>
    ::getNumberOfEquations(kmcd, kmcdSurf);

  const ordinal_type per_team_extent = TChem::TransientContStirredTankReactor::getWorkSpaceSize(
    kmcd,
    kmcdSurf,
    cstr); /// this +m seems to be included in the workspace size calculation

  Kokkos::parallel_for(
    profile_name,
    policy,
    KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
      const ordinal_type i = member.league_rank();
      const RealType1DViewType fac_at_i =
        Kokkos::subview(fac, i, Kokkos::ALL());
      const auto tadv_at_i = tadv(i);
      const RealType1DViewType state_at_i =
        Kokkos::subview(state, i, Kokkos::ALL());
      const RealType0DViewType t_out_at_i = Kokkos::subview(t_out, i);
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
        const real_type t_beg = tadv_at_i._tbeg, t_end = tadv_at_i._tend;

        const real_type temperature = sv_at_i.Temperature();
        const real_type pressure = sv_at_i.Pressure();
        const real_type density = sv_at_i.Density();
        const RealType1DViewType Ys = sv_at_i.MassFractions();

        const RealType0DViewType temperature_out(sv_at_i.TemperaturePtr());
        const RealType0DViewType pressure_out(sv_at_i.PressurePtr());
        const RealType0DViewType density_out(sv_at_i.DensityPtr());
        const RealType1DViewType Ys_out = Ys;

        const RealType1DViewType Zs_out_at_i = Zs_at_i;

        auto wptr = work.data();
        const RealType1DViewType vals(wptr, m);
        wptr += m;
        const RealType1DViewType ww(wptr,
                                    work.extent(0) - (wptr - work.data()));

        TChem::TransientContStirredTankReactor::packToValues(
          member, temperature, Ys, Zs_at_i, vals);

        member.team_barrier();
        TChem::Impl::TransientContStirredTankReactor ::team_invoke(member,
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
                                                   cstr);

        member.team_barrier();
        TChem::TransientContStirredTankReactor::unpackFromValues(member,
                                                 vals,
                                                 temperature_out,
                                                 Ys_out,
                                                 Zs_out_at_i);
      }
    });
  Kokkos::Profiling::popRegion();
}

void
TransientContStirredTankReactor::runDeviceBatch( /// thread block size
  typename UseThisTeamPolicy<exec_space>::type& policy,
  /// input
  const real_type_1d_view& tol_newton,
  const real_type_2d_view& tol_time,
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
  const cstr_data_type& cstr)
{

  TransientContStirredTankReactor_TemplateRun( /// template arguments deduction
    "TChem::TransientContStirredTankReactor::runHostBatch",
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
    /// output
    t_out,
    dt_out,
    state_out,
    Z_out,
    /// const data of kinetic model
    kmcd,
    kmcdSurf,
    cstr);
}

} // namespace TChem
