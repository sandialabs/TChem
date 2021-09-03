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
#include "TChem_EntropyMass.hpp"

namespace TChem {

namespace Impl {
  template<typename PolicyType,
           typename DeviceType>
void
EntropyMass_TemplateRun( /// required template arguments
  const std::string& profile_name,
  /// team size setting
  const PolicyType& policy,
  const Tines::value_type_2d_view<real_type, DeviceType>& state,
  // outputFile
  const Tines::value_type_2d_view<real_type, DeviceType>& EntropyMass,
  const Tines::value_type_1d_view<real_type, DeviceType>& EntropyMixMass,
  const KineticModelConstData<DeviceType >& kmcd)
{
  Kokkos::Profiling::pushRegion(profile_name);
  using policy_type = PolicyType;
  using device_type = DeviceType;

  using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
  using real_type_0d_view_type = Tines::value_type_0d_view<real_type, device_type>;

  using Entropy0SpecMl = Impl::Entropy0SpecMlFcn<real_type,device_type>;

  const ordinal_type level = 1;
  const ordinal_type per_team_extent = EntropyMass::getWorkSpaceSize(kmcd);

  Kokkos::parallel_for(
    profile_name,
    policy,
    KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
      const ordinal_type i = member.league_rank();
      const real_type_1d_view_type state_at_i =
        Kokkos::subview(state, i, Kokkos::ALL());
      const real_type_1d_view_type EntropyMass_at_i =
        Kokkos::subview(EntropyMass, i, Kokkos::ALL());
      const real_type_0d_view_type EntropyMixMass_at_i =
        Kokkos::subview(EntropyMixMass, i);
      Scratch<real_type_1d_view_type> work(member.team_scratch(level),
                                       per_team_extent);
      auto w = (real_type*)work.data();
      auto cpks = real_type_1d_view_type(w, kmcd.nSpec);
      w += kmcd.nSpec;

      const Impl::StateVector<real_type_1d_view_type> sv_at_i(kmcd.nSpec,
                                                          state_at_i);
      TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                        "Error: input state vector is not valid");
      {
        const real_type t = sv_at_i.Temperature();
        const real_type p = sv_at_i.Pressure();
        const real_type_1d_view_type Ys = sv_at_i.MassFractions();

        Entropy0SpecMl::team_invoke(
          member, t, EntropyMass_at_i, cpks, kmcd);

        member.team_barrier();

        //
        real_type sumSk(0);
        Kokkos::parallel_reduce(
          Kokkos::TeamVectorRange(member, kmcd.nSpec),
          [&](const ordinal_type& k, real_type& update) {

            update += EntropyMass_at_i(k) * Ys(k)/kmcd.sMass(k);
            EntropyMass_at_i(k) /=kmcd.sMass(k);
          },
          sumSk);

        EntropyMixMass_at_i() = sumSk;
      }
    });
  Kokkos::Profiling::popRegion();
}

} // namespace Impl

void
EntropyMass::runDeviceBatch( /// thread block size
  const exec_space& exec_space_instance,
  const ordinal_type& team_size,
  const ordinal_type& vector_size,
  /// input
  const ordinal_type& nBatch,
  const real_type_2d_view_type& state,
  /// output
  const real_type_2d_view_type& EntropyMass,
  const real_type_1d_view_type& EntropyMixMass,
  /// const data from kinetic model
  const kinetic_model_type& kmcd)
{

  using policy_type =
    typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;

  const ordinal_type level = 1;
  const ordinal_type per_team_extent = EntropyMass::getWorkSpaceSize(kmcd);
  const ordinal_type per_team_scratch =
    Scratch<real_type_1d_view_type>::shmem_size(per_team_extent);

  policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());
  if (team_size > 0 && vector_size > 0) {
    policy = policy_type(exec_space_instance, nBatch, team_size, vector_size);
  }

  policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

  Impl::EntropyMass_TemplateRun("TChem::EntropyMass::runDeviceBatch",
                                 /// team policy
                                 policy,
                                 state,
                                 EntropyMass,
                                 EntropyMixMass,
                                 kmcd);
}

void
EntropyMass::runDeviceBatch( /// thread block size
  typename UseThisTeamPolicy<exec_space>::type& policy,
  const real_type_2d_view_type& state,
  /// output
  const real_type_2d_view_type& EntropyMass,
  const real_type_1d_view_type& EntropyMixMass,
  /// const data from kinetic model
  const kinetic_model_type& kmcd)
{
  Impl::EntropyMass_TemplateRun("TChem::EntropyMass::runDeviceBatch",
                                 /// team policy
                                 policy,
                                 state,
                                 EntropyMass,
                                 EntropyMixMass,
                                 kmcd);

}

void
EntropyMass::runHostBatch( /// thread block size
  typename UseThisTeamPolicy<host_exec_space>::type& policy,
  const real_type_2d_view_host_type& state,
  /// output
  const real_type_2d_view_host_type& EntropyMass,
  const real_type_1d_view_host_type& EntropyMixMass,
  /// const data from kinetic model
  const kinetic_model_host_type& kmcd)
{
  Impl::EntropyMass_TemplateRun("TChem::EntropyMass::runHostBatch",
                                 /// team policy
                                 policy,
                                 state,
                                 EntropyMass,
                                 EntropyMixMass,
                                 kmcd);

}

} // namespace TChem
