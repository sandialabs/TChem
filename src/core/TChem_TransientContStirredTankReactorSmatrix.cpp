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

#include "TChem_TransientContStirredTankReactorSmatrix.hpp"

namespace TChem {


  template<typename PolicyType,
           typename DeviceType>
  void
  TransientContStirredTankReactorSmatrix_TemplateRun( /// required template arguments
    const std::string& profile_name,
    /// team size setting
    const PolicyType& policy,
    /// input
    const Tines::value_type_2d_view<real_type, DeviceType>& state,
    const Tines::value_type_2d_view<real_type, DeviceType>& site_fraction,
    /// output
    const Tines::value_type_3d_view<real_type, DeviceType>& Smat,
    const Tines::value_type_3d_view<real_type, DeviceType>& Ssmat,
    const Tines::value_type_2d_view<real_type, DeviceType>& Sconv,
    // kinetic model parameters
    const KineticModelConstData<DeviceType >& kmcd,
    const KineticSurfModelConstData<DeviceType>& kmcdSurf,
    const TransientContStirredTankReactorData<DeviceType>& cstr)
  {
    Kokkos::Profiling::pushRegion(profile_name);
    using policy_type = PolicyType;
    using device_type = DeviceType;

    const ordinal_type level = 1;

    const ordinal_type per_team_extent =
     TChem::TransientContStirredTankReactorSmatrix
          ::getWorkSpaceSize(kmcd, kmcdSurf); ///

    //
    using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
    using real_type_2d_view_type = Tines::value_type_2d_view<real_type, device_type>;

    Kokkos::parallel_for(
      profile_name,
      policy,
      KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
        const ordinal_type i = member.league_rank();
        const real_type_1d_view_type state_at_i =
          Kokkos::subview(state, i, Kokkos::ALL());
        // site fraction
        const real_type_1d_view_type Zs_at_i =
          Kokkos::subview(site_fraction, i, Kokkos::ALL());

        const real_type_1d_view_type Sconv_at_i =
        Kokkos::subview(Sconv, i, Kokkos::ALL());

        const real_type_2d_view_type Smat_at_i =
          Kokkos::subview(Smat, i, Kokkos::ALL(), Kokkos::ALL());
        const real_type_2d_view_type Ssmat_at_i =
          Kokkos::subview(Ssmat, i, Kokkos::ALL(), Kokkos::ALL());

        Scratch<real_type_1d_view_type> work(member.team_scratch(level),
                                         per_team_extent);

        Impl::StateVector<real_type_1d_view_type> sv_at_i(kmcd.nSpec, state_at_i);
        TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                          "Error: input state vector is not valid");
        {

          const real_type temperature = sv_at_i.Temperature();
          const real_type pressure = sv_at_i.Pressure();
          const real_type density = sv_at_i.Density();
          const real_type_1d_view_type Ys = sv_at_i.MassFractions();

          Impl::TransientContStirredTankReactorSmatrix<real_type,device_type> ::team_invoke(member,
                                                 temperature,
                                                 Ys,
                                                 Zs_at_i,
                                                 density,
                                                 pressure ,
                                                 Smat_at_i,
                                                 Ssmat_at_i, // output
                                                 Sconv_at_i,
                                                 work,
                                                 kmcd,
                                                 kmcdSurf,
                                                 cstr);


        }
      });
    Kokkos::Profiling::popRegion();
  }

  void
  TransientContStirredTankReactorSmatrix::runDeviceBatch( /// thread block size
    const ordinal_type team_size,
    const ordinal_type vector_size,
    //inputs
    const ordinal_type nBatch,
    const real_type_2d_view_type& state,
    const real_type_2d_view_type& site_fraction,
    /// output
    const real_type_3d_view_type& Smat,
    const real_type_3d_view_type& Ssmat,
    const real_type_2d_view_type& Sconv,
    /// const data from kinetic model
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf,
    const cstr_data_type& cstr)
  {

    // setting policy
    const auto exec_space_instance = TChem::exec_space();
    using policy_type =
      typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;
    const ordinal_type level = 1;
    const ordinal_type per_team_extent =
     TChem::TransientContStirredTankReactorSmatrix
          ::getWorkSpaceSize(kmcd, kmcdSurf); ///
    const ordinal_type per_team_scratch =
        Scratch<real_type_1d_view_type>::shmem_size(per_team_extent);
    policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());

    if (team_size > 0 && vector_size > 0) {
      policy = policy_type(exec_space_instance, nBatch, team_size, vector_size);
    }

    policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

    TransientContStirredTankReactorSmatrix_TemplateRun( /// template arguments deduction
      "TChem::TransientContStirredTankReactorSmatrix::runDeviceBatch",
      /// team policy
      policy,
      //inputs
      state,
      site_fraction,
      //outputs
      Smat,
      Ssmat,
      Sconv,
      /// const data of kinetic model
      kmcd,
      kmcdSurf,
      cstr);
  }

  void
  TransientContStirredTankReactorSmatrix::runDeviceBatch( /// thread block size
    typename UseThisTeamPolicy<exec_space>::type& policy,
    const real_type_2d_view_type& state,
    const real_type_2d_view_type& site_fraction,
    /// output
    const real_type_3d_view_type& Smat,
    const real_type_3d_view_type& Ssmat,
    const real_type_2d_view_type& Sconv,
    /// const data from kinetic model
    const kinetic_model_type& kmcd,
    const kinetic_surf_model_type& kmcdSurf,
    const cstr_data_type& cstr)
  {

    TransientContStirredTankReactorSmatrix_TemplateRun( /// template arguments deduction
      "TChem::TransientContStirredTankReactorSmatrix::runDeviceBatch",
      /// team policy
      policy,
      state,
      site_fraction,
      //outputs
      Smat,
      Ssmat,
      Sconv,
      /// const data of kinetic model
      kmcd,
      kmcdSurf,
      cstr);
  }


  // void
  // TransientContStirredTankReactorSmatrix::runHostBatch( /// thread block size
  //   const ordinal_type team_size,
  //   const ordinal_type vector_size,
  //   //inputs
  //   const ordinal_type nBatch,
  //   const real_type_2d_view_host& state,
  //   const real_type_2d_view_host& zSurf,
  //   /// output
  //   const real_type_3d_view_host& Smat,
  //   const real_type_3d_view_host& Ssmat,
  //   const real_type_2d_view_host& Sconv,
  //   /// const data from kinetic model
  //   const KineticModelConstDataDevice& kmcd,
  //   const KineticSurfModelConstDataDevice& kmcdSurf,
  //   const TransientContStirredTankReactorData& cstr)
  // {
  //
  //   // setting policy
  //   const auto exec_space_instance = TChem::host_exec_space();
  //   using policy_type =
  //     typename TChem::UseThisTeamPolicy<TChem::host_exec_space>::type;
  //   const ordinal_type level = 1;
  //   const ordinal_type per_team_extent =
  //    TChem::TransientContStirredTankReactorSmatrix
  //         ::getWorkSpaceSize(kmcd, kmcdSurf); ///
  //   const ordinal_type per_team_scratch =
  //       Scratch<real_type_1d_view_host>::shmem_size(per_team_extent);
  //   policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());
  //
  //   if (team_size > 0 && vector_size > 0) {
  //     policy = policy_type(exec_space_instance, nBatch, team_size, vector_size);
  //   }
  //
  //   policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
  //
  //   TransientContStirredTankReactorSmatrix_TemplateRun( /// template arguments deduction
  //     "TChem::TransientContStirredTankReactorSmatrix::runHostBatch",
  //     real_type_0d_view_host(),
  //     real_type_1d_view_host(),
  //     /// team policy
  //     policy,
  //     //inputs
  //     state,
  //     zSurf,
  //     //outputs
  //     Smat,
  //     Ssmat,
  //     Sconv,
  //     /// const data of kinetic model
  //     kmcd,
  //     kmcdSurf,
  //     cstr);
  // }
  //
  // void
  // TransientContStirredTankReactorSmatrix::runHostBatch( /// thread block size
  //   typename UseThisTeamPolicy<host_exec_space>::type& policy,
  //   const real_type_2d_view_host& state,
  //   const real_type_2d_view_host& zSurf,
  //   /// output
  //   const real_type_3d_view_host& Smat,
  //   const real_type_3d_view_host& Ssmat,
  //   const real_type_2d_view_host& Sconv,
  //   /// const data from kinetic model
  //   const KineticModelConstDataDevice& kmcd,
  //   const KineticSurfModelConstDataDevice& kmcdSurf,
  //   const TransientContStirredTankReactorData& cstr)
  // {
  //
  //   TransientContStirredTankReactorSmatrix_TemplateRun( /// template arguments deduction
  //     "TChem::TransientContStirredTankReactorSmatrix::runHostBatch",
  //     real_type_0d_view_host(),
  //     real_type_1d_view_host(),
  //     /// team policy
  //     policy,
  //     //inputs
  //     state,
  //     zSurf,
  //     //outputs
  //     Smat,
  //     Ssmat,
  //     Sconv,
  //     /// const data of kinetic model
  //     kmcd,
  //     kmcdSurf,
  //     cstr);
  // }

} // namespace TChem
