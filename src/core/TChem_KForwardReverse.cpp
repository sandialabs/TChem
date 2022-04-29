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
#include "TChem_KForwardReverse.hpp"

namespace TChem {

template <typename PolicyType,
          typename DeviceType>
void KForwardReverse_TemplateRun( /// input
    const std::string &profile_name,
    /// team size setting
    const PolicyType &policy,

    const Tines::value_type_2d_view<real_type, DeviceType> &state,
    /// output
    const Tines::value_type_2d_view<real_type, DeviceType> &kfor,
    const Tines::value_type_2d_view<real_type, DeviceType> &krev,
    /// const data from kinetic model
    const Kokkos::View<KineticModelConstData<DeviceType> *, DeviceType> &kmcds) {
  Kokkos::Profiling::pushRegion(profile_name);
  using policy_type = PolicyType;
  using device_type = DeviceType;
  using Gk = Impl::GkFcn<real_type, device_type>;
  using KForwardReverse = Impl::KForwardReverse<real_type, device_type>;

  using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
  using ordinal_1d_view_type = Tines::value_type_1d_view<ordinal_type, device_type>;

  const ordinal_type level = 1;
  const ordinal_type per_team_extent = TChem::KForwardReverse::getWorkSpaceSize(kmcds(0));

  Kokkos::parallel_for(
      profile_name, policy, KOKKOS_LAMBDA(const typename policy_type::member_type &member) {
        const ordinal_type i = member.league_rank();
        const auto kmcd_at_i = (kmcds.extent(0) == 1 ? kmcds(0) : kmcds(i));
        const real_type_1d_view_type state_at_i = Kokkos::subview(state, i, Kokkos::ALL());
        const real_type_1d_view_type kfor_at_i = Kokkos::subview(kfor, i, Kokkos::ALL());
        //
        const real_type_1d_view_type krev_at_i = Kokkos::subview(krev, i, Kokkos::ALL());

        Scratch<real_type_1d_view_type> work(member.team_scratch(level), per_team_extent);

        const Impl::StateVector<real_type_1d_view_type> sv_at_i(kmcd_at_i.nSpec, state_at_i);
        TCHEM_CHECK_ERROR(!sv_at_i.isValid(), "Error: input state vector is not valid");
        {
          const auto t = sv_at_i.Temperature();
          const auto p = sv_at_i.Pressure();

          auto w = (real_type *)work.data();

          auto gk = real_type_1d_view_type(w, kmcd_at_i.nSpec);
          w += kmcd_at_i.nSpec;
          auto hks = real_type_1d_view_type(w, kmcd_at_i.nSpec);
          w += kmcd_at_i.nSpec;
          auto cpks = real_type_1d_view_type(w, kmcd_at_i.nSpec);
          w += kmcd_at_i.nSpec;

          auto iter = ordinal_1d_view_type((ordinal_type *)w, kmcd_at_i.nReac * 2);
          w += kmcd_at_i.nReac * 2;

          const ordinal_type work_kfor_rev_size =
              Impl::KForwardReverse<real_type, device_type>::getWorkSpaceSize(kmcd_at_i);

          auto work = real_type_1d_view_type(w, work_kfor_rev_size);
          w += work_kfor_rev_size;

          // 1. compute thermo
          Gk::team_invoke(member,
                          t, /// input
                          gk,
                          hks,  /// output
                          cpks, /// workspace
                          kmcd_at_i);

          member.team_barrier();
          // 2. compute rate constant
          KForwardReverse::team_invoke(member, t, p,
                                       gk, /// input
                                       kfor_at_i,
                                       krev_at_i, /// output
                                       iter, work, kmcd_at_i);
        }
      });
  Kokkos::Profiling::popRegion();
}

void KForwardReverse::runDeviceBatch( /// thread block size
    typename UseThisTeamPolicy<exec_space>::type &policy, const real_type_2d_view_type &state,
    const real_type_2d_view_type &kfor, const real_type_2d_view_type &krev, const kinetic_model_type &kmcd) {
  Kokkos::View<kinetic_model_type *, device_type> kmcds(do_not_init_tag("KForwardReverse::kmcds"), 1);
  Kokkos::deep_copy(kmcds, kmcd);

  KForwardReverse_TemplateRun("TChem::KForwardReverse::runDeviceBatch",
                              /// team policy
                              policy, state, kfor, krev, kmcds);
}

void KForwardReverse::runDeviceBatch( /// thread block size
    typename UseThisTeamPolicy<exec_space>::type &policy, const real_type_2d_view_type &state,
    const real_type_2d_view_type &kfor, const real_type_2d_view_type &krev,
    const Kokkos::View<kinetic_model_type *, device_type> &kmcds) {
  KForwardReverse_TemplateRun("TChem::KForwardReverse::runDeviceBatch",
                              /// team policy
                              policy, state, kfor, krev, kmcds);
}

} // namespace TChem
