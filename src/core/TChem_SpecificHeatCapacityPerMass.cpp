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
#include "TChem_SpecificHeatCapacityPerMass.hpp"

namespace TChem {

namespace Impl {
template<typename PolicyType,
         typename RealType0DViewType,
         typename RealType1DViewType,
         typename RealType2DViewType,
         typename KineticModelConstType>
void
SpecificHeatCapacityPerMass_TemplateRun( /// required template arguments
  const std::string& profile_name,
  const RealType0DViewType& dummy_0d,
  /// team size setting
  const PolicyType& policy,
  const RealType2DViewType& state,
  // outputFile
  const RealType2DViewType& CpMass,
  const RealType1DViewType& CpMixMass,
  const KineticModelConstType& kmcd)
{
  Kokkos::Profiling::pushRegion(profile_name);
  using policy_type = PolicyType;

  Kokkos::parallel_for(
    profile_name,
    policy,
    KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
      const ordinal_type i = member.league_rank();
      const RealType1DViewType state_at_i =
        Kokkos::subview(state, i, Kokkos::ALL());
      const RealType1DViewType CpMass_at_i =
        Kokkos::subview(CpMass, i, Kokkos::ALL());
      const RealType0DViewType CpMixMass_at_i = Kokkos::subview(CpMixMass, i);

      const Impl::StateVector<RealType1DViewType> sv_at_i(kmcd.nSpec,
                                                          state_at_i);
      TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                        "Error: input state vector is not valid");
      {
        const real_type t = sv_at_i.Temperature();
        const real_type p = sv_at_i.Pressure();
        const RealType1DViewType Ys = sv_at_i.MassFractions();

        CpMixMass_at_i() =
          Impl::CpMixMs ::team_invoke(member, t, Ys, CpMass_at_i, kmcd);
      }
    });
  Kokkos::Profiling::popRegion();
}

} // namespace Impl

void
SpecificHeatCapacityPerMass::runDeviceBatch( /// thread block size
  typename UseThisTeamPolicy<exec_space>::type& policy,
  const real_type_2d_view& state,
  /// output
  const real_type_2d_view& CpMass,
  const real_type_1d_view& CpMixMass,
  /// const data from kinetic model
  const KineticModelConstDataDevice& kmcd)
{

  Impl::SpecificHeatCapacityPerMass_TemplateRun(
    "TChem::SpecificHeatCapacityPerMass::runDeviceBatch",
    real_type_0d_view(),
    /// team policy
    policy,
    state,
    CpMass,
    CpMixMass,
    kmcd);
}

void
SpecificHeatCapacityPerMass::runHostBatch( /// thread block size
  typename UseThisTeamPolicy<host_exec_space>::type& policy,
  const real_type_2d_view_host& state,
  /// output
  const real_type_2d_view_host& CpMass,
  const real_type_1d_view_host& CpMixMass,
  /// const data from kinetic model
  const KineticModelConstDataHost& kmcd)
{

  Impl::SpecificHeatCapacityPerMass_TemplateRun(
    "TChem::SpecificHeatCapacityPerMass::runDeviceBatch",
    real_type_0d_view_host(),
    /// team policy
    policy,
    state,
    CpMass,
    CpMixMass,
    kmcd);
}

} // namespace TChem
