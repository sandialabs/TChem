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
#include "TChem_InitialCondSurface.hpp"

namespace TChem {

  template<typename PolicyType,
           typename ValueType,
           typename DeviceType>
  void
  InitialCondSurface_TemplateRunModelVariation( /// required template arguments
    const std::string& profile_name,
    const ValueType& dummyValueType,
    /// team size setting
    const PolicyType& policy,
    // inputs
    const Tines::value_type_1d_view<real_type, DeviceType> & tol_newton,
    const real_type & max_num_newton_iterations,
    const Tines::value_type_2d_view<real_type, DeviceType> & fac,
    const Tines::value_type_2d_view<real_type, DeviceType> & state,
    const Tines::value_type_2d_view<real_type, DeviceType> & site_fraction,
    /// output
    const Tines::value_type_2d_view<real_type, DeviceType> & site_fraction_out,
    /// const data from kinetic model
    const Kokkos::View<KineticModelConstData<DeviceType >*,DeviceType>& kmcds,
    const Kokkos::View<KineticSurfModelConstData<DeviceType>*,DeviceType>& kmcdSurfs)
    {
      Kokkos::Profiling::pushRegion(profile_name);
      using policy_type = PolicyType;
      using device_type = DeviceType;
      using value_type = ValueType;

      using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
      using real_type_0d_view_type = Tines::value_type_0d_view<real_type, device_type>;

      const ordinal_type level = 1;
      const ordinal_type per_team_extent = TChem::InitialCondSurface::getWorkSpaceSize(kmcds(0), kmcdSurfs(0));

      Kokkos::parallel_for(
        profile_name,
        policy,
        KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
          const ordinal_type i = member.league_rank();
          const auto kmcd_at_i = (kmcds.extent(0) == 1 ? kmcds(0) : kmcds(i));
          const auto kmcd_surf_at_i = (kmcdSurfs.extent(0) == 1 ? kmcdSurfs(0) : kmcdSurfs(i));
          const real_type_1d_view_type fac_at_i =
            Kokkos::subview(fac, i, Kokkos::ALL());
          const real_type_1d_view_type state_at_i =
            Kokkos::subview(state, i, Kokkos::ALL());
          // site fraction
          const real_type_1d_view_type Zs_at_i =
            Kokkos::subview(site_fraction, i, Kokkos::ALL());

          const real_type_1d_view_type Zs_out_at_i =
            Kokkos::subview(site_fraction_out, i, Kokkos::ALL());

          Scratch<real_type_1d_view_type> work(member.team_scratch(level),
                                          per_team_extent);

          Impl::StateVector<real_type_1d_view_type> sv_at_i(kmcd_at_i.nSpec, state_at_i);
          TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                            "Error: input state vector is not valid");
          {
            const real_type temperature = sv_at_i.Temperature();
            const real_type pressure = sv_at_i.Pressure();
            const real_type_1d_view_type Ys = sv_at_i.MassFractions();

            Impl::InitialCondSurface<value_type, device_type> ::team_invoke(
              member,
              tol_newton,
              max_num_newton_iterations,
              temperature, /// temperature
              Ys,          /// mass fraction (kmcd.nSpec)
              pressure,    /// pressure
              Zs_at_i,
              Zs_out_at_i,
              fac_at_i,
              work, // work
              kmcd_at_i,
              kmcd_surf_at_i);

          }
        });


      Kokkos::Profiling::popRegion();

    }
void
InitialCondSurface::runDeviceBatch( /// input
  const typename TChem::UseThisTeamPolicy<TChem::exec_space>::type& policy,
  const real_type_1d_view_type & tol_newton,
  const real_type& max_num_newton_iterations,
  const real_type_2d_view_type& state,
  const real_type_2d_view_type& site_fraction,
  /// output
  const real_type_2d_view_type& site_fraction_out,
  const real_type_2d_view_type& fac,
  /// const data from kinetic model
  const kinetic_model_type& kmcd,
  const kinetic_surf_model_type& kmcdSurf)
{
  const std::string profile_name ="TChem::InitialCondtionSurface::runDeviceBatch ";

  Kokkos::View<kinetic_model_type*,device_type>
       kmcds(do_not_init_tag("PlugFlowReactor::kmcds"), 1);
   Kokkos::deep_copy(kmcds, kmcd);

 //
 Kokkos::View<kinetic_surf_model_type*,device_type>
      kmcdSurfs(do_not_init_tag("PlugFlowReactor::kmcdSurfs"), 1);
  Kokkos::deep_copy(kmcdSurfs, kmcdSurf);

  InitialCondSurface_TemplateRunModelVariation( /// required template arguments
    profile_name,
    real_type(),
    /// team size setting
    policy,
    // inputs
    tol_newton,
    max_num_newton_iterations,
    fac,
    state,
    site_fraction,
    /// output
    site_fraction_out,
    /// const data from kinetic model
    kmcds,
    kmcdSurfs);
}

void
InitialCondSurface::runDeviceBatch( /// input
  const typename TChem::UseThisTeamPolicy<TChem::exec_space>::type& policy,
  const real_type_1d_view_type & tol_newton,
  const real_type& max_num_newton_iterations,
  const real_type_2d_view_type& state,
  const real_type_2d_view_type& site_fraction,
  /// output
  const real_type_2d_view_type& site_fraction_out,
  const real_type_2d_view_type& fac,
  /// const data from kinetic model
  const Kokkos::View<kinetic_model_type*,device_type>& kmcds,
  const Kokkos::View<kinetic_surf_model_type*,device_type>& kmcdSurfs)
{
  const std::string profile_name ="TChem::InitialCondtionSurface::runDeviceBatch model variation";

  InitialCondSurface_TemplateRunModelVariation( /// required template arguments
    profile_name,
    real_type(),
    /// team size setting
    policy,
    // inputs
    tol_newton,
    max_num_newton_iterations,
    fac,
    state,
    site_fraction,
    /// output
    site_fraction_out,
    /// const data from kinetic model
    kmcds,
    kmcdSurfs);

}

} // namespace TChem
