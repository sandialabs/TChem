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

#include "TChem_Impl_PlugFlowReactorRHS.hpp"
#include "TChem_PlugFlowReactorRHS.hpp"


namespace TChem {

void
PlugFlowReactorRHS::runHostBatch( /// input
  const ordinal_type nBatch,
  /// input
  const real_type_2d_view_host& state,
  const real_type_2d_view_host& zSurf,
  const real_type_1d_view_host& velocity,
  /// output
  const real_type_2d_view_host& rhs,
  /// const data from kinetic model
  const KineticModelConstDataHost& kmcd,
  /// const data from kinetic model
  const KineticSurfModelConstDataHost& kmcdSurf,
  const pfr_data_type& pfrd)
{

  using policy_type = Kokkos::TeamPolicy<host_exec_space>;
  // data for PRF
  const ordinal_type level = 1;
  const ordinal_type per_team_extent = getWorkSpaceSize(kmcd, kmcdSurf);
  const ordinal_type per_team_scratch =
    Scratch<real_type_1d_view>::shmem_size(per_team_extent);

  // policy_type policy(nBatch); // error
  policy_type policy(nBatch, Kokkos::AUTO()); // fine

  // policy_type policy(nBatch, Kokkos::AUTO(), Kokkos::AUTO()); // error
  policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

  PlugFlowReactorRHS_TemplateRun( /// input
    "TChem::PlugFlowReactorRHS::runHostBatch",
    real_type_0d_view_host(),
    /// team size setting
    policy,
    state,
    zSurf,
    velocity,
    rhs,
    kmcd,
    kmcdSurf,
    pfrd);

}

void
PlugFlowReactorRHS::runDeviceBatch( /// input
  const ordinal_type nBatch,
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
  const pfr_data_type& pfrd)
{

  using policy_type = Kokkos::TeamPolicy<exec_space>;
  // data for PRF
  const ordinal_type level = 1;
  const ordinal_type per_team_extent = getWorkSpaceSize(kmcd, kmcdSurf);
  const ordinal_type per_team_scratch =
    Scratch<real_type_1d_view>::shmem_size(per_team_extent);

  policy_type policy(nBatch, Kokkos::AUTO()); // fine
  policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

  PlugFlowReactorRHS_TemplateRun( /// input
    "TChem::PlugFlowReactorRHS::runDeviceBatch",
    real_type_0d_view(),
    /// team size setting
    policy,
    state,
    zSurf,
    velocity,
    rhs,
    kmcd,
    kmcdSurf,
    pfrd);

}

void
PlugFlowReactorRHS::runDeviceBatch( /// input
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
  const pfr_data_type& pfrd)
{

  PlugFlowReactorRHS_TemplateRun( /// input
    "TChem::PlugFlowReactorRHS::runDeviceBatch",
    real_type_0d_view(),
    /// team size setting
    policy,
    state,
    zSurf,
    velocity,
    rhs,
    kmcd,
    kmcdSurf,
    pfrd);

}

void
PlugFlowReactorRHS::runHostBatch( /// input
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
  const pfr_data_type& pfrd)
{

  PlugFlowReactorRHS_TemplateRun( /// input
    "TChem::PlugFlowReactorRHS::runDeviceBatch",
    real_type_0d_view_host(),
    /// team size setting
    policy,
    state,
    zSurf,
    velocity,
    rhs,
    kmcd,
    kmcdSurf,
    pfrd);

}


} // namespace TChem
