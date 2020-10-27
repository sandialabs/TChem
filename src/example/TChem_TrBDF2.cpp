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
#include "TChem_CommandLineParser.hpp"
#include "TChem_Impl_NewtonSolver.hpp"
#include "TChem_Impl_TrBDF2.hpp"
#include "TChem_Util.hpp"

namespace TChem {
namespace Example {

struct TestProblemTrBDF
{
  using real_type_1d_view_type = real_type_1d_view;
  using real_type_2d_view_type = real_type_2d_view;

  KOKKOS_DEFAULTED_FUNCTION
  TestProblemTrBDF() = default;

  /// system of ODEs
  /// dx1/dt = -20 x1 - 0.25 x2 - 19.75 x3
  /// dx2/dt = 20 x1 - 20.25 x2 + 0.25 x3
  /// dx3/dt = 20 x1 - 19.75 x2 - 0.25 x3
  /// T = [0, 10], x(0) = (1, 0, -1)^T
  /// exact solution is
  /// x1(t) =  1/2(exp(-0.5t) + exp(-20t)(cos 20t + sin 20t))
  /// x2(t) =  1/2(exp(-0.5t) - exp(-20t)(cos 20t - sin 20t))
  /// x3(t) = -1/2(exp(-0.5t) + exp(-20t)(cos 20t - sin 20t))

  KOKKOS_INLINE_FUNCTION
  ordinal_type getNumberOfTimeODEs() const { return 3; }

  KOKKOS_INLINE_FUNCTION
  ordinal_type getNumberOfConstraints() const { return 0; }

  KOKKOS_INLINE_FUNCTION
  ordinal_type getNumberOfEquations() const
  {
    return getNumberOfTimeODEs() + getNumberOfConstraints();
  }

  template<typename MemberType, typename RealType1DViewType>
  KOKKOS_INLINE_FUNCTION void computeInitValues(
    const MemberType& member,
    const RealType1DViewType& x) const
  {
    /// do nothing; we use solution from the previous timestep
  }

  template<typename MemberType,
           typename RealType1DViewType,
           typename RealType2DViewType>
  KOKKOS_INLINE_FUNCTION void computeJacobian(const MemberType& member,
                                              const RealType1DViewType& x,
                                              const RealType2DViewType& J) const
  {
    Kokkos::single(Kokkos::PerTeam(member), [&]() {
      J(0, 0) = -20.0;
      J(0, 1) = -0.25;
      J(0, 2) = -19.75;

      J(1, 0) = 20.0;
      J(1, 1) = -20.25;
      J(1, 2) = 0.25;

      J(2, 0) = 20.0;
      J(2, 1) = -19.75;
      J(2, 2) = -0.25;
    });
    member.team_barrier();
  }

  template<typename MemberType, typename RealType1DViewType>
  KOKKOS_INLINE_FUNCTION void computeFunction(const MemberType& member,
                                              const RealType1DViewType& x,
                                              const RealType1DViewType& f) const
  {
    Kokkos::single(Kokkos::PerTeam(member), [&]() {
      const real_type x0 = x(0), x1 = x(1), x2 = x(2);
      f(0) = -20 * x0 - 0.25 * x1 - 19.75 * x2;
      f(1) = 20 * x0 - 20.25 * x1 + 0.25 * x2;
      f(2) = 20 * x0 - 19.75 * x1 - 0.25 * x2;
    });
    member.team_barrier();
  }

  template<typename MemberType, typename RealType1DViewType>
  KOKKOS_INLINE_FUNCTION real_type computeError(const MemberType& member,
                                                const real_type& t,
                                                const RealType1DViewType& x)
  {
    const real_type x0 =
      0.5 * (ats<real_type>::exp(-0.5 * t) +
             ats<real_type>::exp(-20.0 * t) *
               (ats<real_type>::cos(20.0 * t) + ats<real_type>::sin(20.0 * t)));
    const real_type x1 =
      0.5 * (ats<real_type>::exp(-0.5 * t) -
             ats<real_type>::exp(-20.0 * t) *
               (ats<real_type>::cos(20.0 * t) - ats<real_type>::sin(20.0 * t)));
    const real_type x2 =
      -0.5 * (ats<real_type>::exp(-0.5 * t) +
              ats<real_type>::exp(-20.0 * t) * (ats<real_type>::cos(20.0 * t) -
                                                ats<real_type>::sin(20.0 * t)));

    const real_type abs_x0 = ats<real_type>::abs(x0 - x(0));
    const real_type abs_x1 = ats<real_type>::abs(x1 - x(1));
    const real_type abs_x2 = ats<real_type>::abs(x2 - x(2));

    const real_type err_norm =
      ats<real_type>::sqrt(abs_x0 * abs_x0 + abs_x1 * abs_x1 + abs_x2 * abs_x2);
    const real_type sol_norm =
      ats<real_type>::sqrt(x0 * x0 + x1 * x1 + x2 * x2);
    const real_type rel_norm = err_norm / sol_norm / 3.0;
    Kokkos::single(Kokkos::PerTeam(member),
                   [&]() { printf("time %e, err %e \n", t, rel_norm); });
    return rel_norm;
  }
};
} // namespace Example
} // namespace TChem

using ordinal_type = TChem::ordinal_type;
using real_type = TChem::real_type;
using real_type_1d_view = TChem::real_type_1d_view;
using real_type_2d_view = TChem::real_type_2d_view;
using real_type_3d_view = TChem::real_type_3d_view;

int
main(int argc, char* argv[])
{

  /// default inputs
  int nBatch(1);
  bool verbose(true);

  /// parse command line arguments
  TChem::CommandLineParser opts(
    "This example computes reaction rates with a given state vector");
  opts.set_option<int>(
    "batchsize",
    "Batchsize the same state vector described in statefile is cloned",
    &nBatch);
  opts.set_option<bool>(
    "verbose", "If true, printout the first Jacobian values", &verbose);

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse)
    return 0; // print help return

  Kokkos::initialize(argc, argv);
  {
    const bool detail = false;

    TChem::exec_space::print_configuration(std::cout, detail);
    TChem::host_exec_space::print_configuration(std::cout, detail);

    {
      const ordinal_type uDim = 3;
      /// input with initial values zero
      real_type_2d_view u("u", nBatch, uDim);

      Kokkos::Impl::Timer timer;

      timer.reset();
      using policy_type = Kokkos::TeamPolicy<TChem::exec_space>;
      const ordinal_type level = 0, per_team_extent = 57;
      const ordinal_type per_team_scratch =
        TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);

      policy_type policy(nBatch, Kokkos::AUTO());
      policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

      Kokkos::parallel_for(
        "TChem TrBDF2 Example",
        policy,
        KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
          const ordinal_type i = member.league_rank();
          TChem::Example::TestProblemTrBDF problem;

          TChem::Impl::TrBDF2<TChem::exec_space> trbdf;
          TChem::Impl::TrBDF2_Part1<TChem::Example::TestProblemTrBDF>
            trbdf_part1;
          TChem::Impl::TrBDF2_Part2<TChem::Example::TestProblemTrBDF>
            trbdf_part2;

          /// batch solution
          auto u_at_i = real_type_1d_view(&u(i, 0), 3);

          /// initial condition
          Kokkos::single(Kokkos::PerTeam(member), [&]() {
            u_at_i(0) = 1;
            u_at_i(1) = 0;
            u_at_i(2) = -1;
          });
          member.team_barrier();

          /// set time step
          const real_type tbeg(0), tend(10);
          const real_type dtmin = (tend - tbeg) / real_type(10000);
          const real_type dtmax = (tend - tbeg) / real_type(10);
          const real_type two(2),
            gamma = two - TChem::ats<real_type>::sqrt(two);

          trbdf._gamma = gamma;
          trbdf_part1._gamma = gamma;
          trbdf_part2._gamma = gamma;

          real_type dt = dtmin;

          /// use scratch space
          auto work = TChem::Scratch<real_type_1d_view>(
            member.team_scratch(level), per_team_extent);
          auto wptr = work.data();

          /// trbdf workspace
          auto un = real_type_1d_view(wptr, 3);
          wptr += 3;
          auto unr = real_type_1d_view(wptr, 3);
          wptr += 3;
          auto fn = real_type_1d_view(wptr, 3);
          wptr += 3;
          auto fnr = real_type_1d_view(wptr, 3);
          wptr += 3;

          /// newton workspace
          auto dx = real_type_1d_view(wptr, 3);
          wptr += 3;
          auto f = real_type_1d_view(wptr, 3);
          wptr += 3;
          auto J = real_type_2d_view(wptr, 3, 3);
          wptr += 9;
          auto tol_time = real_type_2d_view(wptr, 3, 2);
          wptr += 3;
          // auto w  = real_type_1d_view(wptr, 27); wptr += 27;
          const ordinal_type newton_workspace_size =
            TChem::Impl::NewtonSolver::getWorkSpaceSize(problem);
          auto w = real_type_1d_view(wptr, newton_workspace_size);
          wptr += newton_workspace_size;

          /// compute initial un and unr
          const real_type tol = 1.e-6;
          Kokkos::parallel_for(Kokkos::TeamVectorRange(member, 3),
                               [&](const ordinal_type& k) {
                                 un(k) = u_at_i(k);
                                 unr(k) = u_at_i(k);
                                 tol_time(k, 0) = 0;
                                 tol_time(k, 1) = tol;
                               });
          member.team_barrier();
          const ordinal_type max_time_integration = 100000;
          const real_type zero(0);
          real_type t(0);
          for (ordinal_type titer = 0;
               titer < max_time_integration && dt != zero;
               ++titer, t += dt) {
            trbdf_part1._dt = dt;
            trbdf_part2._dt = dt;

            /// part 1
            {
              /// set the views into trbdf
              trbdf_part1._un = un;
              trbdf_part1._fn = fn;

              problem.computeFunction(member, un, fn);

              /// simple setting for newton
              real_type atol = 1e-6, rtol = 1e-5;
              ordinal_type /* m = 3, */ max_iter = 10, iter_count = 0,
                                        converge = 0;

              /// run the newton iterations
              TChem::Impl::NewtonSolver ::team_invoke(member,
                                                      trbdf_part1,
                                                      atol,
                                                      rtol,
                                                      max_iter, // input
                                                      unr,      // input/output
                                                      dx,
                                                      f,
                                                      J,
                                                      w, // workspace
                                                      iter_count,
                                                      converge); // output

              problem.computeFunction(member, unr, fnr);

              Kokkos::single(Kokkos::PerTeam(member), [&]() {
                if (i == 0) {
                  if (converge) {
                    // printf("Test problem part 1 converged with a solution %e
                    // %e %e\n",
                    //       unr(0), unr(1), unr(2));
                  } else {
                    printf("Test problem part 1 failed to converge with "
                           "iteration count %d\n",
                           iter_count);
                  }
                }
              });
            }

            /// part 2
            {
              /// set the views into trbdf
              trbdf_part2._un = un;
              trbdf_part2._unr = unr;

              /// simple setting for newton
              real_type atol = 1e-6, rtol = 1e-5;
              ordinal_type /* m = 3, */ max_iter = 10, iter_count = 0,
                                        converge = 0;

              /// run the newton iterations
              TChem::Impl::NewtonSolver ::team_invoke(member,
                                                      trbdf_part2,
                                                      atol,
                                                      rtol,
                                                      max_iter, // input
                                                      u_at_i,   // input/output
                                                      dx,
                                                      f,
                                                      J,
                                                      w, // workspace
                                                      iter_count,
                                                      converge); // output

              problem.computeFunction(member, u_at_i, f);

              Kokkos::single(Kokkos::PerTeam(member), [&]() {
                if (i == 0) {
                  if (converge) {
                    // printf("Test problem part 2 converged with a solution %e
                    // %e %e\n",
                    //       u_at_i(0), u_at_i(1), u_at_i(2));
                  } else {
                    printf("Test problem part 2 failed to converge with "
                           "iteration count %d\n",
                           iter_count);
                  }
                }
              });
            }

            /// part 3
            {
              problem.computeError(member, t, u_at_i);
              Kokkos::parallel_for(
                Kokkos::TeamVectorRange(member, 3),
                [&](const ordinal_type& k) { un(k) = u_at_i(k); });
              const ordinal_type m = 3;
              trbdf.computeTimeStepSize(
                member, dtmin, dtmax, tol_time, m, fn, fnr, f, u_at_i, dt);
              if ((t + dt) > tend)
                dt = tend - t;
            }
          }
        });
      Kokkos::fence(); /// timing purpose
      const real_type t_trbdf2 = timer.seconds();

      printf("Time for TrBDF2 %e sec\n", t_trbdf2);
    }
  }
  Kokkos::finalize();

  return 0;
}
