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


#include "TChem_PlugFlowReactorNumJacobian.hpp"

namespace TChem {

namespace Impl {

template<typename PolicyType,
         typename RealType0DViewType,
         typename RealType1DViewType,
         typename RealType2DViewType,
         typename RealType3DViewType,
         typename KineticModelConstType,
         typename KineticSurfModelConstData,
         typename PlugFlowReactorConstDataType>
void
PlugFlowReactorNumJacobian_TemplateRun( /// required template arguments
  const std::string& profile_name,
  const RealType0DViewType& dummy_0d,
  /// team size setting
  const PolicyType& policy,
  const RealType2DViewType& state,
  const RealType2DViewType& zSurf,
  const RealType1DViewType& velocity,
  // outputFile
  const RealType3DViewType& jac,
  const RealType2DViewType& fac,
  const KineticModelConstType& kmcd,
  const KineticSurfModelConstData& kmcdSurf,
  const PlugFlowReactorConstDataType& pfrd)
{
  Kokkos::Profiling::pushRegion(profile_name);
  using policy_type = PolicyType;

  const ordinal_type level = 1;
  const ordinal_type per_team_extent = PlugFlowReactorNumJacobian::getWorkSpaceSize(kmcd, kmcdSurf, pfrd);

  const ordinal_type m = kmcd.nSpec + 3 + kmcdSurf.nSpec;

  Kokkos::parallel_for(
    profile_name,
    policy,
    KOKKOS_LAMBDA(const typename policy_type::member_type& member) {

      const ordinal_type i = member.league_rank();
      const RealType1DViewType state_at_i =
        Kokkos::subview(state, i, Kokkos::ALL());

      const RealType2DViewType jac_at_i =
        Kokkos::subview(jac, i, Kokkos::ALL(), Kokkos::ALL());
      //
      const RealType1DViewType fac_at_i =
        Kokkos::subview(fac, i, Kokkos::ALL());

      Scratch<RealType1DViewType> work(member.team_scratch(level),
                                       per_team_extent);

      const Impl::StateVector<RealType1DViewType> sv_at_i(kmcd.nSpec,
                                                          state_at_i);
      TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                        "Error: input state vector is not valid");
      {
        const real_type t = sv_at_i.Temperature();
        const real_type p = sv_at_i.Pressure();
        const RealType1DViewType Ys = sv_at_i.MassFractions();
        const real_type density = sv_at_i.Density();
        const real_type vel_at_i = velocity(i);
        // site fraction
        const RealType1DViewType Zs_at_i = Kokkos::subview(zSurf, i, Kokkos::ALL());

        auto wptr = work.data();
        const RealType1DViewType vals(wptr, m);
        wptr += m;
        const RealType1DViewType ww(wptr,
                                    work.extent(0) - (wptr - work.data()));

        TChem::PlugFlowReactor::packToValues(
          member, t, Ys, density, vel_at_i, Zs_at_i, vals);
        member.team_barrier();

        Impl::PlugFlowReactorNumJacobian::team_invoke(member,
                                                vals,
                                                jac_at_i,
                                                fac_at_i, // output
                                                ww,
                                                kmcd,
                                                kmcdSurf,
                                                pfrd);
      }
    });
  Kokkos::Profiling::popRegion();
}

} // namespace Impl

void
PlugFlowReactorNumJacobian::runDeviceBatch( /// thread block size
  typename UseThisTeamPolicy<exec_space>::type& policy,
  const real_type_2d_view& state,
  /// output
  const real_type_2d_view& zSurf,
  const real_type_1d_view& velocity,
  // outputFile
  const real_type_3d_view& jac,
  const real_type_2d_view& fac,
  const KineticModelConstDataDevice& kmcd,
  const KineticSurfModelConstDataDevice& kmcdSurf,
  const pfr_data_type& pfrd)
{
  Impl::PlugFlowReactorNumJacobian_TemplateRun(
    "TChem::PlugFlowReactorNumJacobian::runDeviceBatch",
                                 real_type_0d_view(),
                                 /// team policy
                                 policy,
                                 /// input
                                 state,
                                 zSurf,
                                 velocity,
                                 //outputs
                                 jac,
                                 fac,
                                 //const parameters
                                 kmcd,
                                 kmcdSurf,
                                 pfrd);
}

void
PlugFlowReactorNumJacobian::runDeviceBatch( /// thread block size
  const ordinal_type& team_size,
  const ordinal_type& vector_size,
  const ordinal_type nBatch,
  const real_type_2d_view& state,
  /// output
  const real_type_2d_view& zSurf,
  const real_type_1d_view& velocity,
  // outputFile
  const real_type_3d_view& jac,
  const real_type_2d_view& fac,
  const KineticModelConstDataDevice& kmcd,
  const KineticSurfModelConstDataDevice& kmcdSurf,
  const pfr_data_type& pfrd)
{
  // hard coded parameter for PFR: I need to fix it OD
  // const auto pfrd = PlugFlowReactorRHS::createConstData<exec_space>();

  const auto exec_space_instance = TChem::exec_space();
  using policy_type =
    typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;

  const ordinal_type level = 1;
  const ordinal_type per_team_extent = getWorkSpaceSize(kmcd, kmcdSurf, pfrd);

  const ordinal_type per_team_scratch =
    Scratch<real_type_1d_view>::shmem_size(per_team_extent);

  policy_type policy(nBatch, Kokkos::AUTO()); // fine
  if (team_size > 0 && vector_size > 0) {
    policy = policy_type(exec_space_instance, nBatch, team_size, vector_size);
  }

  policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));


  Impl::PlugFlowReactorNumJacobian_TemplateRun(
    "TChem::PlugFlowReactorNumJacobian::runDeviceBatch",
                                 real_type_0d_view(),
                                 /// team policy
                                 policy,
                                 /// input
                                 state,
                                 zSurf,
                                 velocity,
                                 //outputs
                                 jac,
                                 fac,
                                 //const parameters
                                 kmcd,
                                 kmcdSurf,
                                 pfrd);
}


} // namespace TChem
