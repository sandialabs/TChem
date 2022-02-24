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

#include "TChem_IsothermalTransientContStirredTankReactorRHS.hpp"

namespace TChem {


  template<typename PolicyType,
           typename DeviceType>
  void
  IsothermalTransientContStirredTankReactorRHS_TemplateRunModelVariation( /// required template arguments
    const std::string& profile_name,
    const PolicyType& policy,

    // inputs
    const Tines::value_type_2d_view<real_type, DeviceType>& state,
    const Tines::value_type_2d_view<real_type, DeviceType>& site_fraction,
    /// output
    /// const data from kinetic model
    const Tines::value_type_2d_view<real_type, DeviceType>& rhs,
    const Kokkos::View<KineticModelConstData<DeviceType >*,DeviceType>& kmcds,
    const Kokkos::View<KineticSurfModelConstData<DeviceType>*,DeviceType>& kmcdSurfs,
    const TransientContStirredTankReactorData<DeviceType>& cstr)
  {
    Kokkos::Profiling::pushRegion(profile_name);
    using policy_type = PolicyType;
    using device_type = DeviceType;

    using real_type_0d_view_type = Tines::value_type_0d_view<real_type, device_type>;
    using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
    using real_type_2d_view_type = Tines::value_type_2d_view<real_type, device_type>;

    using IsothermalTransientContStirredTankReactorRHS =
    Impl::IsothermalTransientContStirredTankReactorRHS<real_type, device_type>;

    const ordinal_type level = 1;

    const ordinal_type per_team_extent =
     TChem::IsothermalTransientContStirredTankReactorRHS
          ::getWorkSpaceSize(kmcds(0), kmcdSurfs(0)); ///
    //
    const ordinal_type per_team_scratch =
        Scratch<real_type_1d_view_type>::shmem_size(per_team_extent);

    Kokkos::parallel_for(
      profile_name,
      policy,
      KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
        const ordinal_type i = member.league_rank();
        const auto kmcd_at_i = (kmcds.extent(0) == 1 ? kmcds(0) : kmcds(i));
        const auto kmcd_surf_at_i = (kmcdSurfs.extent(0) == 1 ? kmcdSurfs(0) : kmcdSurfs(i));
        const real_type_1d_view_type state_at_i =
          Kokkos::subview(state, i, Kokkos::ALL());
        // site fraction
        const real_type_1d_view_type site_fraction_at_i =
          Kokkos::subview(site_fraction, i, Kokkos::ALL());

        const real_type_1d_view_type rhs_at_i =
        Kokkos::subview(rhs, i, Kokkos::ALL());

        Scratch<real_type_1d_view_type> work(member.team_scratch(level),
                                         per_team_extent + 1);
        //
        real_type* wptr = work.data();
        const real_type_1d_view_type mass_flow_out_work(wptr,1);
        wptr += 1;
        const real_type_0d_view_type mass_flow_out = Kokkos::subview(mass_flow_out_work, 0);;
        const real_type_1d_view_type ww(wptr,
                                    work.extent(0) - (wptr - work.data()));


        Impl::StateVector<real_type_1d_view_type> sv_at_i(kmcd_at_i.nSpec, state_at_i);
        TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                          "Error: input state vector is not valid");
        {




          const real_type temperature = sv_at_i.Temperature();
          const real_type pressure = sv_at_i.Pressure();
          const real_type_1d_view_type Ys = sv_at_i.MassFractions();
          // when density is lower than zero, tchem will compute it.
          const real_type density = sv_at_i.Density() > 0 ? sv_at_i.Density() :
                            Impl::RhoMixMs<real_type,device_type>
                                ::team_invoke(member,  sv_at_i.Temperature(),
                                sv_at_i.Pressure(),
                                sv_at_i.MassFractions(), kmcd_at_i);
          member.team_barrier();
          IsothermalTransientContStirredTankReactorRHS::team_invoke(member,
                                                 temperature,
                                                 Ys,
                                                 site_fraction_at_i,
                                                 density,
                                                 pressure,
                                                 rhs_at_i,
                                                 mass_flow_out,
                                                 ww,
                                                 kmcd_at_i,
                                                 kmcd_surf_at_i,
                                                 cstr);




        }
      });
    Kokkos::Profiling::popRegion();
  }

  template<typename PolicyType,
           typename DeviceType>
  void
  IsothermalTransientContStirredTankReactorRHS_TemplateRun( /// required template arguments
    const std::string& profile_name,
    const PolicyType& policy,

    // inputs
    const Tines::value_type_2d_view<real_type, DeviceType>& state,
    const Tines::value_type_2d_view<real_type, DeviceType>& site_fraction,
    /// output
    /// const data from kinetic model
    const Tines::value_type_2d_view<real_type, DeviceType>& rhs,
    const KineticModelConstData<DeviceType >& kmcd,
    const KineticSurfModelConstData<DeviceType>& kmcdSurf,
    const TransientContStirredTankReactorData<DeviceType>& cstr)
  {
    using kinetic_model_type = KineticModelConstData<DeviceType>;
    using kinetic_surf_model_type = KineticSurfModelConstData<DeviceType>;

    Kokkos::View<kinetic_model_type*,DeviceType>
         kmcds(do_not_init_tag("isothermalTCSTR::kmcds"), 1);
     Kokkos::deep_copy(kmcds, kmcd);
    Kokkos::View<kinetic_surf_model_type*,DeviceType>
        kmcdSurfs(do_not_init_tag("isothermalTCSTR::kmcdSurfs"), 1);
    Kokkos::deep_copy(kmcdSurfs, kmcdSurf);
    IsothermalTransientContStirredTankReactorRHS_TemplateRunModelVariation(profile_name,
       policy, state, site_fraction, rhs,  kmcds, kmcdSurfs, cstr   );

  }

  void
  IsothermalTransientContStirredTankReactorRHS::runDeviceBatch( /// thread block size
    const ordinal_type nBatch,
    const real_type_2d_view_type& state,
    const real_type_2d_view_type& site_fraction,
    /// output
    const real_type_2d_view_type& rhs,
    /// const data from kinetic model
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf,
    const cstr_data_type& cstr)
  {

    const auto exec_space_instance = TChem::exec_space();
    using policy_type =
      typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;
    const ordinal_type level = 1;
    const ordinal_type per_team_extent =
     TChem::IsothermalTransientContStirredTankReactorRHS
          ::getWorkSpaceSize(kmcd, kmcdSurf); ///

    const ordinal_type per_team_scratch =
        Scratch<real_type_1d_view_type>::shmem_size(per_team_extent);

    policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());
    policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

    IsothermalTransientContStirredTankReactorRHS_TemplateRun( /// template arguments deduction
      "TChem::IsothermalTransientContStirredTankReactorRHS::runDeviceBatch",
      policy,
      state,
      site_fraction,
      rhs,
      /// const data of kinetic model
      kmcd,
      kmcdSurf,
      cstr);
  }

  void
  IsothermalTransientContStirredTankReactorRHS::runDeviceBatch( /// thread block size
    typename UseThisTeamPolicy<exec_space>::type& policy,
    const real_type_2d_view_type& state,
    const real_type_2d_view_type& site_fraction,
    /// output
    const real_type_2d_view_type& rhs,
    /// const data from kinetic model
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf,
    const cstr_data_type& cstr)
  {
    IsothermalTransientContStirredTankReactorRHS_TemplateRun( /// template arguments deduction
      "TChem::IsothermalTransientContStirredTankReactorRHS::runDeviceBatch",
      policy,
      state,
      site_fraction,
      rhs,
      /// const data of kinetic model
      kmcd,
      kmcdSurf,
      cstr);
  }

  void
  IsothermalTransientContStirredTankReactorRHS::runDeviceBatch( /// thread block size
    typename UseThisTeamPolicy<exec_space>::type& policy,
    const real_type_2d_view_type& state,
    const real_type_2d_view_type& site_fraction,
    /// output
    const real_type_2d_view_type& rhs,
    /// const data from kinetic model
    const Kokkos::View<kinetic_model_type*,device_type>& kmcds,
    const Kokkos::View<kinetic_surf_model_type*,device_type>& kmcdSurfs,
    const cstr_data_type& cstr)
  {
    IsothermalTransientContStirredTankReactorRHS_TemplateRunModelVariation( /// template arguments deduction
      "TChem::IsothermalTransientContStirredTankReactorRHS_ModelVariation::runDeviceBatch",
      policy,
      state,
      site_fraction,
      rhs,
      /// const data of kinetic model
      kmcds,
      kmcdSurfs,
      cstr);
  }

} // namespace TChem
