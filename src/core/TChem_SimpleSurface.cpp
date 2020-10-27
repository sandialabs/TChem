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
#include "TChem_SimpleSurface.hpp"
#include "TChem_Impl_SimpleSurface.hpp"
#include "TChem_Util.hpp"

namespace TChem {

void
SimpleSurface::runDeviceBatch( /// input
  typename UseThisTeamPolicy<exec_space>::type& policy,
  const real_type_1d_view& tol_newton,
  const real_type_2d_view& tol_time,
  const time_advance_type_1d_view& tadv,
  const real_type_2d_view& state,
  const real_type_2d_view& zSurf,
  /// output
  const real_type_1d_view& t_out,
  const real_type_1d_view& dt_out,
  const real_type_2d_view& Z_out,
  const real_type_2d_view& fac, // jac comput
  /// const data from kinetic model
  const KineticModelConstDataDevice& kmcd,
  const KineticSurfModelConstDataDevice& kmcdSurf)
{
  Kokkos::Profiling::pushRegion("TChem::SimpleSurface::runDeviceBatch");
  using policy_type = typename UseThisTeamPolicy<exec_space>::type;

  const ordinal_type level = 1;
  // const ordinal_type m =
  // Impl::SimpleSurface_Problem<KineticModelConstDataDevice,
  //                        KineticSurfModelConstDataDevice>::getProblemDimension(kmcdSurf);
  const ordinal_type per_team_extent = getWorkSpaceSize(kmcd, kmcdSurf);
  //  const ordinal_type per_team_scratch =
  //  Scratch<real_type_1d_view>::shmem_size(per_team_extent);

  // policy_type policy(nBatch, Kokkos::AUTO());
  // policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

  Kokkos::parallel_for(
    "TChem::SimpleSurface::runDeviceBatch",
    policy,
    KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
      const ordinal_type i = member.league_rank();
      const real_type_1d_view fac_at_i =
        Kokkos::subview(fac, i, Kokkos::ALL());
      const time_advance_type tadv_at_i = tadv(i);
      const real_type_1d_view state_at_i =
        Kokkos::subview(state, i, Kokkos::ALL());
      const real_type_0d_view t_out_at_i = Kokkos::subview(t_out, i);
      const real_type_0d_view dt_out_at_i = Kokkos::subview(dt_out, i);
      // site fraction
      const real_type_1d_view Zs_at_i =
        Kokkos::subview(zSurf, i, Kokkos::ALL());

      Scratch<real_type_1d_view> work(member.team_scratch(level),
                                      per_team_extent);

      // using problem_type =
      //   Impl::SimpleSurface_Problem<KineticModelConstDataDevice,
      //                               KineticSurfModelConstDataDevice>;
      // real_type* wptr = work.data();
      // real_type_2d_view tol_time(
      //   wptr, problem_type::getNumberOfEquations(kmcdSurf), 2);
      // wptr += tol_time.span();
      // real_type_1d_view tol_newton(wptr, 2);
      // wptr += tol_newton.span();

      // real_type_1d_view ww(wptr, work.extent(0) - (wptr - work.data()));

      Impl::StateVector<real_type_1d_view> sv_at_i(kmcd.nSpec, state_at_i);
      TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                        "Error: input state vector is not valid");
      {
        const ordinal_type max_num_newton_iterations =
          tadv_at_i._max_num_newton_iterations;
        const ordinal_type max_num_time_iterations =
          tadv_at_i._num_time_iterations_per_interval;

        const real_type dt_in = tadv_at_i._dt, dt_min = tadv_at_i._dtmin,
                        dt_max = tadv_at_i._dtmax;
        const real_type t_beg = tadv_at_i._tbeg, t_end = tadv_at_i._tend;

        const real_type temperature = sv_at_i.Temperature();
        const real_type pressure = sv_at_i.Pressure();
        const real_type_1d_view Ys = sv_at_i.MassFractions();

        const real_type_1d_view Zs_out_at_i = Zs_at_i;

        // const real_type atol_newton = 1e-12, rtol_newton = 1e-8,
        //                 tol_time_rel = 1e-4, tol_time_abs = 1e-12;
        //
        // {
        //   Kokkos::parallel_for(
        //     Kokkos::TeamVectorRange(member, tol_time.extent(0)),
        //     [&](const ordinal_type& i) {
        //       tol_time(i, 0) = tol_time_abs;
        //       tol_time(i, 1) = tol_time_rel;
        //       if (i == 0) {
        //         tol_newton(0) = atol_newton;
        //         tol_newton(1) = rtol_newton;
        //       }
        //     });
        //   member.team_barrier();
        // }

        Impl::SimpleSurface ::team_invoke(member,
                                          max_num_newton_iterations,
                                          max_num_time_iterations,
                                          tol_newton,
                                          tol_time,
                                          fac_at_i,
                                          dt_in,
                                          dt_min,
                                          dt_max,
                                          t_beg,
                                          t_end,
                                          Zs_at_i,
                                          t_out_at_i,
                                          dt_out_at_i,
                                          Zs_at_i,
                                          temperature, /// temperature
                                          pressure,    /// pressure
                                          Ys, /// mass fraction (kmcd.nSpec)
                                          work, // work
                                          kmcd,
                                          kmcdSurf);
      }
    });
  Kokkos::Profiling::popRegion();
}

} // namespace TChem
