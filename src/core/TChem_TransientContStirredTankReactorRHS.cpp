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

#include "TChem_TransientContStirredTankReactorRHS.hpp"

namespace TChem {


  template<typename RealType0DViewType,
           typename RealType1DViewType,
           typename RealType2DViewType,
           typename KineticModelConstType,
           typename KineticSurfModelConstData,
           typename TransientContStirredTankReactorConstDataType>
  void
  TransientContStirredTankReactorRHS_TemplateRun( /// required template arguments
    const std::string& profile_name,
    const RealType0DViewType& dummy_0d,
    const RealType1DViewType& dummy_1d,
    const ordinal_type nBatch,

    // inputs
    const RealType2DViewType& state,
    const RealType2DViewType& zSurf,
    /// output
    /// const data from kinetic model
    const RealType2DViewType& rhs,
    const KineticModelConstType& kmcd,
    const KineticSurfModelConstData& kmcdSurf,
    const TransientContStirredTankReactorConstDataType& cstr)
  {
    Kokkos::Profiling::pushRegion(profile_name);
    using policy_type = Kokkos::TeamPolicy<exec_space>;

    const ordinal_type level = 1;

    const ordinal_type per_team_extent =
     TChem::TransientContStirredTankReactorRHS
          ::getWorkSpaceSize(kmcd, kmcdSurf); ///

    const ordinal_type per_team_scratch =
        Scratch<RealType1DViewType>::shmem_size(per_team_extent);

    policy_type policy(nBatch, Kokkos::AUTO()); // fine
    policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

    Kokkos::parallel_for(
      profile_name,
      policy,
      KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
        const ordinal_type i = member.league_rank();
        const RealType1DViewType state_at_i =
          Kokkos::subview(state, i, Kokkos::ALL());
        // site fraction
        const RealType1DViewType Zs_at_i =
          Kokkos::subview(zSurf, i, Kokkos::ALL());

        const RealType1DViewType rhs_at_i =
        Kokkos::subview(rhs, i, Kokkos::ALL());

        Scratch<RealType1DViewType> work(member.team_scratch(level),
                                         per_team_extent);

        Impl::StateVector<RealType1DViewType> sv_at_i(kmcd.nSpec, state_at_i);
        TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                          "Error: input state vector is not valid");
        {

          const real_type temperature = sv_at_i.Temperature();
          const real_type pressure = sv_at_i.Pressure();
          const real_type density = sv_at_i.Density();
          const RealType1DViewType Ys = sv_at_i.MassFractions();

          Impl::TransientContStirredTankReactorRHS ::team_invoke(member,
                                                 temperature,
                                                 Ys,
                                                 Zs_at_i,
                                                 density,
                                                 pressure ,
                                                 rhs_at_i,
                                                 work,
                                                 kmcd,
                                                 kmcdSurf,
                                                 cstr);




        }
      });
    Kokkos::Profiling::popRegion();
  }

  void
  TransientContStirredTankReactorRHS::runDeviceBatch( /// thread block size
    const ordinal_type nBatch,
    const real_type_2d_view& state,
    const real_type_2d_view& zSurf,
    /// output
    const real_type_2d_view& rhs,
    /// const data from kinetic model
    const KineticModelConstDataDevice& kmcd,
    const KineticSurfModelConstDataDevice& kmcdSurf,
    const cstr_data_type& cstr)
  {
    TransientContStirredTankReactorRHS_TemplateRun( /// template arguments deduction
      "TChem::TransientContStirredTankReactorRHS::runHostBatch",
      real_type_0d_view(),
      real_type_1d_view(),
      nBatch,
      state,
      zSurf,
      rhs,
      /// const data of kinetic model
      kmcd,
      kmcdSurf,
      cstr);
  }

} // namespace TChem
