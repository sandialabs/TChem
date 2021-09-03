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
#include "TChem_SimpleSurface.hpp"
#include "TChem_Impl_SimpleSurface.hpp"
#include "TChem_Util.hpp"

namespace TChem {

  template<typename PolicyType,
           typename ValueType,
           typename DeviceType,
           typename TimeAdvance1DViewType>
  void
  SimpleSurface_TemplateRunModelVariation(
    const std::string& profile_name,
    const ValueType& dummyValueType,
    /// team size setting
    const PolicyType& policy,
    // inputs
    const Tines::value_type_1d_view<real_type, DeviceType> & tol_newton,
    const Tines::value_type_2d_view<real_type, DeviceType> & tol_time,
    const TimeAdvance1DViewType& tadv,
    const Tines::value_type_2d_view<real_type, DeviceType> & state,
    const Tines::value_type_2d_view<real_type, DeviceType> & site_fraction,
    /// output
    const Tines::value_type_1d_view<real_type, DeviceType> & t_out,
    const Tines::value_type_1d_view<real_type, DeviceType> & dt_out,
    const Tines::value_type_2d_view<real_type, DeviceType> & site_fraction_out,
    const Tines::value_type_2d_view<real_type, DeviceType> & fac,
    /// const data from kinetic model
    const Kokkos::View<KineticModelConstData<DeviceType >*,DeviceType>& kmcds,
    const Kokkos::View<KineticSurfModelConstData<DeviceType>*,DeviceType>& kmcdSurfs)
{
  Kokkos::Profiling::pushRegion("TChem::SimpleSurface::runDeviceBatch");
  using policy_type = PolicyType;
  using device_type = DeviceType;
  using value_type = ValueType;
  using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
  using real_type_0d_view_type = Tines::value_type_0d_view<real_type, device_type>;
  const ordinal_type level = 1;
  const ordinal_type per_team_extent = SimpleSurface::getWorkSpaceSize(kmcds(0), kmcdSurfs(0));

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
      // site fraction
      const real_type_1d_view_type Zs_at_i =
        Kokkos::subview(site_fraction, i, Kokkos::ALL());

      Scratch<real_type_1d_view_type> work(member.team_scratch(level),
                                      per_team_extent);

      auto wptr = work.data();
      const real_type_1d_view_type w(wptr,work.extent(0));


      Impl::StateVector<real_type_1d_view_type> sv_at_i(kmcd_at_i.nSpec, state_at_i);
      TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                        "Error: input state vector is not valid");
      {
        const ordinal_type jacobian_interval =
          tadv_at_i._jacobian_interval;
        const ordinal_type max_num_newton_iterations =
          tadv_at_i._max_num_newton_iterations;
        const ordinal_type max_num_time_iterations =
          tadv_at_i._num_time_iterations_per_interval;

        const real_type dt_in = tadv_at_i._dt, dt_min = tadv_at_i._dtmin,
                        dt_max = tadv_at_i._dtmax;
        const real_type t_beg = tadv_at_i._tbeg, t_end = tadv_at_i._tend;

        const real_type temperature = sv_at_i.Temperature();
        const real_type pressure = sv_at_i.Pressure();
        const real_type_1d_view_type Ys = sv_at_i.MassFractions();

        const real_type_1d_view_type Zs_out_at_i = Zs_at_i;


        Impl::SimpleSurface<value_type, device_type> ::team_invoke(member,
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
                                          Zs_at_i,
                                          t_out_at_i,
                                          dt_out_at_i,
                                          Zs_at_i,
                                          temperature, /// temperature
                                          pressure,    /// pressure
                                          Ys, /// mass fraction (kmcd.nSpec)
                                          w, // work
                                          kmcd_at_i,
                                          kmcd_surf_at_i);
      }
    });
  Kokkos::Profiling::popRegion();
}

void
SimpleSurface::runDeviceBatch( /// input
  typename UseThisTeamPolicy<exec_space>::type& policy,
  const real_type_1d_view_type& tol_newton,
  const real_type_2d_view_type& tol_time,
  const time_advance_type_1d_view& tadv,
  const real_type_2d_view_type& state,
  const real_type_2d_view_type& site_fraction,
  /// output
  const real_type_1d_view_type& t_out,
  const real_type_1d_view_type& dt_out,
  const real_type_2d_view_type& site_fraction_out,
  const real_type_2d_view_type& fac, // jac comput
  /// const data from kinetic model
  const kinetic_model_type& kmcd,
  const kinetic_surf_model_type& kmcdSurf)

{

  Kokkos::View<kinetic_model_type*,device_type>
       kmcds(do_not_init_tag("PlugFlowReactor::kmcds"), 1);
   Kokkos::deep_copy(kmcds, kmcd);

 //
 Kokkos::View<kinetic_surf_model_type*,device_type>
      kmcdSurfs(do_not_init_tag("PlugFlowReactor::kmcdSurfs"), 1);
  Kokkos::deep_copy(kmcdSurfs, kmcdSurf);

  SimpleSurface_TemplateRunModelVariation( /// template arguments deduction
    "TChem::SimpleSurface::runDeviceBatch",
    real_type(),
    /// team policy
    policy,
    /// input
    tol_newton,
    tol_time,
    tadv,
    state,
    site_fraction,
    /// output
    t_out,
    dt_out,
    site_fraction_out,
    fac,
    /// const data of kinetic model
    kmcds,
    kmcdSurfs);
}


void
SimpleSurface::runDeviceBatch( /// input
  typename UseThisTeamPolicy<exec_space>::type& policy,
const real_type_1d_view_type& tol_newton,
const real_type_2d_view_type& tol_time,
const time_advance_type_1d_view& tadv,
const real_type_2d_view_type& state,
const real_type_2d_view_type& site_fraction,
/// output
const real_type_1d_view_type& t_out,
const real_type_1d_view_type& dt_out,
const real_type_2d_view_type& site_fraction_out,
const real_type_2d_view_type& fac, // jac comput
/// const data from kinetic model
const Kokkos::View<kinetic_model_type*,device_type>& kmcds,
const Kokkos::View<kinetic_surf_model_type*,device_type>& kmcdSurfs)
{

  SimpleSurface_TemplateRunModelVariation( /// template arguments deduction
    "TChem::SimpleSurface::runDeviceBatch Model Variation",
    real_type(),
    /// team policy
    policy,
    /// input
    tol_newton,
    tol_time,
    tadv,
    state,
    site_fraction,
    /// output
    t_out,
    dt_out,
    site_fraction_out,
    fac,
    /// const data of kinetic model
    kmcds,
    kmcdSurfs);

 }

} // namespace TChem
