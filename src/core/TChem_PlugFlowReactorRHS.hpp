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
#ifndef __TCHEM_PlugFlowReactorRHS_HPP__
#define __TCHEM_PlugFlowReactorRHS_HPP__

#include "TChem_Util.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_Impl_PlugFlowReactorRHS.hpp"
#include "TChem_PlugFlowReactor.hpp"

using pfr_data_type = TChem::pfr_data_type;

namespace TChem {

struct PlugFlowReactorRHS
{

  template<typename KineticModelConstDataType,
           typename KineticSurfModelConstDataType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd,
    const KineticSurfModelConstDataType& kmcdSurf)
  {
    return (7 * kmcd.nSpec + 8 * kmcd.nReac + 5 * kmcdSurf.nSpec +
            6 * kmcdSurf.nReac);
  }


  template<typename PolicyType,
           typename RealType0DViewType,
           typename RealType1DViewType,
           typename RealType2DViewType,
           typename KineticModelConstType,
           typename KineticSurfModelConstData,
           typename PlugFlowReactorConstDataType>
  static void
  PlugFlowReactorRHS_TemplateRun( /// input
    const std::string& profile_name,
    const RealType0DViewType& dummy_0d,
    /// team size setting
    const PolicyType& policy,
    const RealType2DViewType& state,
    const RealType2DViewType& zSurf,
    const RealType1DViewType& velocity,
    const RealType2DViewType& rhs,
    const KineticModelConstType& kmcd,
    const KineticSurfModelConstData& kmcdSurf,
    const PlugFlowReactorConstDataType& pfrd)
    {
      Kokkos::Profiling::pushRegion(profile_name);
      using policy_type = PolicyType;

      const ordinal_type level = 1;
      const ordinal_type per_team_extent = TChem::PlugFlowReactorRHS
                                                ::getWorkSpaceSize(kmcd, kmcdSurf);

      Kokkos::parallel_for(
        profile_name,
        policy,
        KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
          const ordinal_type i = member.league_rank();
          const RealType1DViewType state_at_i =
            Kokkos::subview(state, i, Kokkos::ALL());
          const RealType0DViewType velocity_at_i =
            Kokkos::subview(velocity, i);

          const RealType1DViewType rhs_at_i = Kokkos::subview(rhs, i, Kokkos::ALL());
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
            const RealType1DViewType Xc = sv_at_i.MassFractions();
            const real_type vel = velocity_at_i();
            // site fraction
            const RealType1DViewType Zs = Kokkos::subview(zSurf, i, Kokkos::ALL());
            Impl::PlugFlowReactorRHS ::team_invoke(member,
                                                   t,
                                                   Xc,
                                                   Zs,
                                                   density,
                                                   p,
                                                   vel,
                                                   rhs_at_i,
                                                   work,
                                                   kmcd,
                                                   kmcdSurf,
                                                   pfrd);
          }
        });
      Kokkos::Profiling::popRegion();
    }

  static void runHostBatch( /// input
    const ordinal_type nBatch,
    const real_type_2d_view_host& state,
    /// input
    const real_type_2d_view_host& zSurf,
    // prf aditional variable
    const real_type_1d_view_host& velocity,
    /// output
    const real_type_2d_view_host& rhs,
    /// const data from kinetic model
    const KineticModelConstDataHost& kmcd,
    /// const data from kinetic model surface
    const KineticSurfModelConstDataHost& kmcdSurf,
    //pfr parameters
    const pfr_data_type& pfrd);

  static void runDeviceBatch( /// input
    const ordinal_type nBatch,
    /// input gas state
    const real_type_2d_view& state,
    /// surface state
    const real_type_2d_view& zSurf,
    // prf aditional variable
    const real_type_1d_view& velocity,
    /// output
    const real_type_2d_view& rhs,
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
    const real_type_2d_view& rhs,
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
    const real_type_2d_view_host& rhs,
    /// const data from kinetic model
    const KineticModelConstDataHost& kmcd,
    /// const data from kinetic model surface
    const KineticSurfModelConstDataHost& kmcdSurf,
    const pfr_data_type& pfrd);
};

} // namespace TChem

#endif
