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

#include "TChem_Impl_RateOfProgressInd.hpp"

#include "TChem_RateOfProgress.hpp"

namespace TChem {

void
RateOfProgress::runDeviceBatch( /// input
  const ordinal_type nBatch,
  const real_type_2d_view& state,
  /// output
  const real_type_2d_view& RoPFor,
  const real_type_2d_view& RoPRev,
  /// const data from kinetic model
  const KineticModelConstDataDevice& kmcd)
{
  Kokkos::Profiling::pushRegion("TChem::RateOfProgress::runDeviceBatch");
  using policy_type = Kokkos::TeamPolicy<exec_space>;

  const ordinal_type level = 1;
  const ordinal_type per_team_extent = getWorkSpaceSize(kmcd);
  const ordinal_type per_team_scratch =
    Scratch<real_type_1d_view>::shmem_size(per_team_extent);

  // policy_type policy(nBatch); // error
  policy_type policy(nBatch, Kokkos::AUTO()); // fine
  // policy_type policy(nBatch, Kokkos::AUTO(), Kokkos::AUTO()); // error
  policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
  Kokkos::parallel_for(
    "TChem::RateOfProgress::runDeviceBatch",
    policy,
    KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
      const ordinal_type i = member.league_rank();
      const real_type_1d_view state_at_i =
        Kokkos::subview(state, i, Kokkos::ALL());
      const real_type_1d_view RoPFor_at_i =
        Kokkos::subview(RoPFor, i, Kokkos::ALL());
      const real_type_1d_view RoPRev_at_i =
        Kokkos::subview(RoPRev, i, Kokkos::ALL());

      Scratch<real_type_1d_view> work(member.team_scratch(level),
                                      per_team_extent);

      const Impl::StateVector<real_type_1d_view> sv_at_i(kmcd.nSpec,
                                                         state_at_i);
      TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                        "Error: input state vector is not valid");
      {
        const real_type t = sv_at_i.Temperature();
        const real_type p = sv_at_i.Pressure();
        const real_type_1d_view Ys = sv_at_i.MassFractions();

        Impl::RateOfProgressInd ::team_invoke(
          member, t, p, Ys, RoPFor_at_i, RoPRev_at_i, work, kmcd);
      }
    });
  Kokkos::Profiling::popRegion();
}

} // namespace TChem
