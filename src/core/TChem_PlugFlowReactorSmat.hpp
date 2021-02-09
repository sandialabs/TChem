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


#ifndef __TCHEM_PLUGFLOWREACTORSMAT_HPP__
#define __TCHEM_PLUGFLOWREACTORSMAT_HPP__

#include "TChem_Impl_PlugFlowReactorSmat.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_PlugFlowReactorRHS.hpp"
#include "TChem_Util.hpp"

namespace TChem {

struct PlugFlowReactorSmat
{

  template<typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf)
  {
    return (2 * kmcd.nSpec + kmcdSurf.nReac);
  }

  template<typename PolicyType,
           typename RealType1DViewType,
           typename RealType2DViewType,
           typename RealType3DViewType,
           typename KineticModelConstType,
           typename KineticSurfModelConstData,
           typename PlugFlowReactorConstDataType>
  static void
  PlugFlowReactorSmat_TemplateRun( /// input
    const std::string& profile_name,
    /// team size setting
    const PolicyType& policy,
    const RealType2DViewType& state,
    const RealType2DViewType& zSurf,
    const RealType1DViewType& velocity,
    const RealType3DViewType& Smat,
    const RealType3DViewType& Ssmat,
    const KineticModelConstType& kmcd,
    const KineticSurfModelConstData& kmcdSurf,
    const PlugFlowReactorConstDataType& pfrd)
    {

      Kokkos::Profiling::pushRegion(profile_name);
      using policy_type = PolicyType;

      const ordinal_type level = 1;
      const ordinal_type per_team_extent = TChem::
      PlugFlowReactorSmat::getWorkSpaceSize(kmcd, kmcdSurf);

      Kokkos::parallel_for(
        profile_name,
        policy,
        KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
          const ordinal_type i = member.league_rank();
          const RealType1DViewType state_at_i =
            Kokkos::subview(state, i, Kokkos::ALL());
          // const real_type_0d_view velocity_at_i = Kokkos::subview(velocity, i);

          const RealType2DViewType Smat_at_i =
            Kokkos::subview(Smat, i, Kokkos::ALL(), Kokkos::ALL());
          const RealType2DViewType Ssmat_at_i =
            Kokkos::subview(Ssmat, i, Kokkos::ALL(), Kokkos::ALL());

          Scratch<RealType1DViewType> work(member.team_scratch(level),
                                          per_team_extent);

          const Impl::StateVector<RealType1DViewType> sv_at_i(kmcd.nSpec,
                                                             state_at_i);
          TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                            "Error: input state vector is not valid");
          {
            const real_type t = sv_at_i.Temperature();
            const real_type p = sv_at_i.Pressure();
            const real_type density = sv_at_i.Density();
            const RealType1DViewType Ys = sv_at_i.MassFractions();
            const real_type vel = velocity(i);
            // site fraction
            const RealType1DViewType Zs = Kokkos::subview(zSurf, i, Kokkos::ALL());
            Impl::PlugFlowReactorSmat ::team_invoke(member,
                                                    t,
                                                    Ys,
                                                    Zs,
                                                    density,
                                                    p,
                                                    vel, // input
                                                    Smat_at_i,
                                                    Ssmat_at_i, // output
                                                    work,
                                                    kmcd,
                                                    kmcdSurf,
                                                    pfrd);
          }
        });
      Kokkos::Profiling::popRegion();
    }


  static void runDeviceBatch( /// input
    const ordinal_type nBatch,
    /// input gas state
    const real_type_2d_view& state,
    /// surface state
    const real_type_2d_view& zSurf,
    // prf aditional variable
    const real_type_1d_view& velocity,
    /// output
    const real_type_3d_view& Smat,
    const real_type_3d_view& Ssmat,
    /// const data from kinetic model
    const KineticModelConstDataDevice& kmcd,
    /// const data from kinetic model surface
    const KineticSurfModelConstDataDevice& kmcdSurf,
    const pfr_data_type& pfrd);

  //
  static void runDeviceBatch( /// input
    typename UseThisTeamPolicy<exec_space>::type& policy,
    const real_type_2d_view& state,
    /// input
    const real_type_2d_view& zSurf,
    const real_type_1d_view& velocity,

    /// output
    const real_type_3d_view& Smat,
    const real_type_3d_view& Ssmat,
    /// const data from kinetic model
    const KineticModelConstDataDevice& kmcd,
    /// const data from kinetic model surface
    const KineticSurfModelConstDataDevice& kmcdSurf,
    const pfr_data_type& pfrd);
  //
  static void runHostBatch( /// input
    typename UseThisTeamPolicy<host_exec_space>::type& policy,
    const real_type_2d_view_host& state,
    /// input
    const real_type_2d_view_host& zSurf,
    const real_type_1d_view_host& velocity,

    /// output
    const real_type_3d_view_host& Smat,
    const real_type_3d_view_host& Ssmat,
    /// const data from kinetic model
    const KineticModelConstDataHost& kmcd,
    /// const data from kinetic model surface
    const KineticSurfModelConstDataHost& kmcdSurf,
    const pfr_data_type& pfrd);
};

} // namespace TChem

#endif
