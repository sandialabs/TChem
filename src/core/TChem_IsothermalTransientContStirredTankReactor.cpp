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

#include "TChem_IsothermalTransientContStirredTankReactor.hpp"
#include "TChem_Impl_IsothermalTransientContStirredTankReactorRHS.hpp"
#include "TChem_Impl_RhoMixMs.hpp"

namespace TChem {

template<typename PolicyType,
         typename ValueType,
         typename DeviceType,
         typename TimeAdvance1DViewType>
void
IsothermalTransientContStirredTankReactor_TemplateRunModelVariation( /// required template arguments
  const std::string& profile_name,
  const ValueType& dummyValueType,
  /// team size setting
  const PolicyType& policy,
  // inputs
  const Tines::value_type_1d_view<real_type, DeviceType> & tol_newton,
  const Tines::value_type_2d_view<real_type, DeviceType> & tol_time,
  const Tines::value_type_2d_view<real_type, DeviceType> & fac,
  const TimeAdvance1DViewType& tadv,
  const Tines::value_type_2d_view<real_type, DeviceType>& state,
  const Tines::value_type_2d_view<real_type, DeviceType>& site_fraction,
  /// output
  const Tines::value_type_1d_view<real_type, DeviceType>& t_out,
  const Tines::value_type_1d_view<real_type, DeviceType>& dt_out,
  const Tines::value_type_2d_view<real_type, DeviceType>& state_out,
  const Tines::value_type_2d_view<real_type, DeviceType>& site_fraction_out,
  const Tines::value_type_1d_view<real_type, DeviceType>& mass_flow_out,

  /// const data from kinetic model
  const Kokkos::View<KineticModelConstData<DeviceType >*,DeviceType>& kmcds,
  const Kokkos::View<KineticSurfModelConstData<DeviceType>*,DeviceType>& kmcdSurfs,
  const TransientContStirredTankReactorData<DeviceType>& cstr)
{
  Kokkos::Profiling::pushRegion(profile_name);
  using policy_type = PolicyType;
  using device_type = DeviceType;
  using value_type = ValueType;
  using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
  using real_type_0d_view_type = Tines::value_type_0d_view<real_type, device_type>;

  const ordinal_type level = 1;

  const ordinal_type m = Impl::IsothermalTransientContStirredTankReactor_Problem<real_type,device_type>
    ::getNumberOfEquations(kmcds(0), kmcdSurfs(0));

  const ordinal_type per_team_extent = TChem::IsothermalTransientContStirredTankReactor::getWorkSpaceSize(
    kmcds(0),
    kmcdSurfs(0)); /// this +m seems to be included in the workspace size calculation

  Kokkos::parallel_for(
    profile_name,
    policy,
    KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
      const ordinal_type i = member.league_rank();
      const auto kmcd_at_i = (kmcds.extent(0) == 1 ? kmcds(0) : kmcds(i));
      const auto kmcd_surf_at_i = (kmcdSurfs.extent(0) == 1 ? kmcdSurfs(0) : kmcdSurfs(i));

      const real_type_1d_view_type fac_at_i =
        Kokkos::subview(fac, i, Kokkos::ALL());
      const auto tadv_at_i = tadv(i);
      const real_type_1d_view_type state_at_i =
        Kokkos::subview(state, i, Kokkos::ALL());
      const real_type_0d_view_type t_out_at_i = Kokkos::subview(t_out, i);
      const real_type_0d_view_type dt_out_at_i = Kokkos::subview(dt_out, i);
      const real_type_0d_view_type mass_flow_out_at_i = Kokkos::subview(mass_flow_out, i);
      // site fraction
      const real_type_1d_view_type Zs_at_i =
        Kokkos::subview(site_fraction, i, Kokkos::ALL());

      Scratch<real_type_1d_view_type> work(member.team_scratch(level),
                                       per_team_extent);

      Impl::StateVector<real_type_1d_view_type> sv_at_i(kmcd_at_i.nSpec, state_at_i);
      TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                        "Error: input state vector is not valid");
      {

        const real_type t_end = tadv_at_i._tend;

      if (t_out_at_i() < t_end) {
        const ordinal_type jacobian_interval = tadv_at_i._jacobian_interval;
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
        const real_type_1d_view_type Ys = sv_at_i.MassFractions();

        const real_type_0d_view_type temperature_out(sv_at_i.TemperaturePtr());
        const real_type_0d_view_type pressure_out(sv_at_i.PressurePtr());
        const real_type_0d_view_type density_out(sv_at_i.DensityPtr());
        const real_type_1d_view_type Ys_out = Ys;

        const real_type_1d_view_type Zs_out_at_i = Zs_at_i;

        auto wptr = work.data();
        const real_type_1d_view_type vals(wptr, m);
        wptr += m;
        const real_type_1d_view_type ww(wptr,
                                    work.extent(0) - (wptr - work.data()));

        TChem::IsothermalTransientContStirredTankReactor::packToValues(
          member, Ys, Zs_at_i, vals);

        member.team_barrier();
        TChem::Impl::IsothermalTransientContStirredTankReactor<value_type,device_type> ::team_invoke(member,
                                                                                           jacobian_interval,
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
                                                   temperature,
                                                   vals,
                                                   t_out_at_i,
                                                   dt_out_at_i,
                                                   vals,
                                                   mass_flow_out_at_i,
                                                   ww, // work
                                                   kmcd_at_i,
                                                   kmcd_surf_at_i,
                                                   cstr);

        member.team_barrier();
        TChem::IsothermalTransientContStirredTankReactor::unpackFromValues(member,
                                                 vals,
                                                 Ys_out,
                                                 Zs_out_at_i);
        // update density with out data
        member.team_barrier();
        density_out() = Impl::RhoMixMs<real_type, device_type>
        ::team_invoke(member, temperature_out(), pressure, Ys_out, kmcd_at_i);
        pressure_out() = pressure; // pressure is constant
        temperature_out() = temperature;// constant

      }
    }
    });
  Kokkos::Profiling::popRegion();
}


void
IsothermalTransientContStirredTankReactor::runDeviceBatch( /// thread block size
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
  const real_type_1d_view_type& mass_flow_out,
  /// const data from kinetic model
  const kinetic_model_type& kmcd,
  const kinetic_surf_model_type& kmcdSurf,
  const TransientContStirredTankReactorData<device_type>& cstr)

{

#define TCHEM_RUN_TRANSIENT_CONT_STIRRED_TANK_REACTOR()           \
  IsothermalTransientContStirredTankReactor_TemplateRunModelVariation(                      \
    profile_name,       \
    value_type(),                                                   \
    policy,                                                         \
    tol_newton,                                                     \
    tol_time,                                                       \
    fac,                                                            \
    tadv,                                                           \
    state,                                                          \
    site_fraction,                                                  \
    t_out,                                                          \
    dt_out,                                                         \
    state_out,                                                      \
    site_fraction_out,                                               \
    mass_flow_out,                                                           \
    kmcds,                                                           \
    kmcdSurfs,                                                       \
    cstr);                                                          \

//
Kokkos::View<kinetic_model_type*,device_type>
     kmcds(do_not_init_tag("CSTR::kmcds"), 1);
 Kokkos::deep_copy(kmcds, kmcd);

//
Kokkos::View<kinetic_surf_model_type*,device_type>
    kmcdSurfs(do_not_init_tag("CSTR::kmcdSurfs"), 1);
Kokkos::deep_copy(kmcdSurfs, kmcdSurf);

const std::string profile_name ="TChem::IsothermalTransientContStirredTankReactor::runDeviceBatch";

#if defined(TCHEM_ENABLE_SACADO_JACOBIAN_TRANSIENT_CONT_STIRRED_TANK_REACTOR)
 using problem_type = Impl::IsothermalTransientContStirredTankReactor_Problem<real_type, device_type>;
 const ordinal_type m = problem_type::getNumberOfEquations(kmcd, kmcdSurf);

 if (m < 128) {
   using value_type = Sacado::Fad::SLFad<real_type,128>;
   TCHEM_RUN_TRANSIENT_CONT_STIRRED_TANK_REACTOR()
 } else if  (m < 256) {
   using value_type = Sacado::Fad::SLFad<real_type,256>;
   TCHEM_RUN_TRANSIENT_CONT_STIRRED_TANK_REACTOR()
 } else if  (m < 512) {
   using value_type = Sacado::Fad::SLFad<real_type,512>;
   TCHEM_RUN_TRANSIENT_CONT_STIRRED_TANK_REACTOR()
 } else if (m < 1024){
   using value_type = Sacado::Fad::SLFad<real_type,1024>;
   TCHEM_RUN_TRANSIENT_CONT_STIRRED_TANK_REACTOR()
 } else{
   TCHEM_CHECK_ERROR(0,
                     "Error: Number of equations is bigger than size of sacado fad type");
 }
#else
  using value_type = real_type;
  TCHEM_RUN_TRANSIENT_CONT_STIRRED_TANK_REACTOR()
#endif
}

void
IsothermalTransientContStirredTankReactor::runDeviceBatch( /// thread block size
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
  const real_type_1d_view_type& mass_flow_out,
  /// const data from kinetic model
  const Kokkos::View<kinetic_model_type*,device_type>& kmcds,
  const Kokkos::View<kinetic_surf_model_type*,device_type>& kmcdSurfs,
  const TransientContStirredTankReactorData<device_type>& cstr)
{

  const std::string profile_name ="TChem::IsothermalTransientContStirredTankReactor::runDeviceBatch Model Variation";

  #if defined(TCHEM_ENABLE_SACADO_JACOBIAN_TRANSIENT_CONT_STIRRED_TANK_REACTOR)
   using problem_type = Impl::IsothermalTransientContStirredTankReactor_Problem<real_type, device_type>;
   const ordinal_type m = problem_type::getNumberOfEquations(kmcds(0), kmcdSurfs(0));

   if (m < 128) {
     using value_type = Sacado::Fad::SLFad<real_type,128>;
     TCHEM_RUN_TRANSIENT_CONT_STIRRED_TANK_REACTOR()
   } else if  (m < 256) {
     using value_type = Sacado::Fad::SLFad<real_type,256>;
     TCHEM_RUN_TRANSIENT_CONT_STIRRED_TANK_REACTOR()
   } else if  (m < 512) {
     using value_type = Sacado::Fad::SLFad<real_type,512>;
     TCHEM_RUN_TRANSIENT_CONT_STIRRED_TANK_REACTOR()
   } else if (m < 1024){
     using value_type = Sacado::Fad::SLFad<real_type,1024>;
     TCHEM_RUN_TRANSIENT_CONT_STIRRED_TANK_REACTOR()
   } else{
     TCHEM_CHECK_ERROR(0,
                       "Error: Number of equations is bigger than size of sacado fad type");
   }
  #else
    using value_type = real_type;
    TCHEM_RUN_TRANSIENT_CONT_STIRRED_TANK_REACTOR()
  #endif

}

} // namespace TChem
