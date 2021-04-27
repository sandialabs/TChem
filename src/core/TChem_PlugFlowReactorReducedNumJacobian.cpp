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
#include "TChem_PlugFlowReactorReducedNumJacobian.hpp"
#include "TChem_Util.hpp"
#include "TChem_PlugFlowReactor.hpp"

namespace TChem {

void
PlugFlowReactorReducedNumJacobian::runDeviceBatch( /// input
  const ordinal_type nBatch,
  const real_type_2d_view& state,
  /// input
  const real_type_2d_view& zSurf,
  const real_type_1d_view& velocity,

  /// output
  const real_type_3d_view& jac,
  const real_type_2d_view& fac,
  /// const data from kinetic model
  const KineticModelConstDataDevice& kmcd,
  /// const data from kinetic model surface
  const KineticSurfModelConstDataDevice& kmcdSurf,
  const pfr_data_type& pfrd)
{

  Kokkos::Profiling::pushRegion("TChem::PlugFlowReactorReducedNumJacobian::runDeviceBatch");
  using policy_type = Kokkos::TeamPolicy<exec_space>;

  // data for PRF
  // const auto pfrd = PlugFlowReactorRHS::createConstData<exec_space>();

  const ordinal_type m = kmcd.nSpec + 3 + kmcdSurf.nSpec;

  // const ordinal_type m = Impl::PlugFlowReactor_Problem<
  //   KineticModelConstType,
  //   KineticSurfModelConstData,
  //   PlugFlowReactorConstDataType>::getNumberOfEquations(kmcd, kmcdSurf);

  const ordinal_type level = 1;
  const ordinal_type per_team_extent = getWorkSpaceSize(kmcd, kmcdSurf, pfrd);

  const ordinal_type per_team_scratch =
    Scratch<real_type_1d_view>::shmem_size(per_team_extent);

  // policy_type policy(nBatch); // error
  policy_type policy(nBatch, Kokkos::AUTO()); // fine
  // policy_type policy(nBatch, Kokkos::AUTO(), Kokkos::AUTO()); // error
  policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
  Kokkos::parallel_for(
    "TChem::PlugFlowReactorReducedNumJacobian::runDeviceBatch",
    policy,
    KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
      const ordinal_type i = member.league_rank();
      const real_type_1d_view state_at_i =
        Kokkos::subview(state, i, Kokkos::ALL());
      // const real_type_0d_view velocity_at_i = Kokkos::subview(velocity, i);

      const real_type_2d_view jac_at_i =
        Kokkos::subview(jac, i, Kokkos::ALL(), Kokkos::ALL());
      //
      const real_type_1d_view fac_at_i =
        Kokkos::subview(fac, i, Kokkos::ALL());

      Scratch<real_type_1d_view> work(member.team_scratch(level),
                                      per_team_extent);

      const Impl::StateVector<real_type_1d_view> sv_at_i(kmcd.nSpec,
                                                         state_at_i);
      TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                        "Error: input state vector is not valid");
      {
        const real_type t = sv_at_i.Temperature();
        const real_type p = sv_at_i.Pressure();
        const real_type density = sv_at_i.Density();
        const real_type_1d_view Ys = sv_at_i.MassFractions();
        const real_type vel_at_i = velocity(i);
        // site fraction
        const real_type_1d_view Zs_at_i = Kokkos::subview(zSurf, i, Kokkos::ALL());

        auto wptr = work.data();
        const real_type_1d_view vals(wptr, m);
        wptr += m;
        const real_type_1d_view ww(wptr,
                                    work.extent(0) - (wptr - work.data()));

        TChem::PlugFlowReactor::packToValues(
          member, t, Ys, density, vel_at_i, Zs_at_i, vals);
        member.team_barrier();

        Impl::PlugFlowReactorReducedNumJacobian::team_invoke(member,
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

} // namespace TChem
