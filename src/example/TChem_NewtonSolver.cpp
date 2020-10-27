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
#include "TChem_Util.hpp"

namespace TChem {
namespace Example {

struct TestProblemNewton
{
  KOKKOS_DEFAULTED_FUNCTION
  TestProblemNewton() = default;

  /// system of equations f
  /// 3 x0 - cos(x1 x2) - 3/2 = 0
  /// 4x0^2 - 625 x1^2 + 2x2 - 1 = 0
  /// 20 x2 + exp(-x0 x1) + 9 = 0
  /// jacobian
  /// J = [ 3 , x2 sin(x1 x2), x1 sin(x1x2) ]
  ///     [ 8 x0 , -1250 x1, 2 ]
  ///     [ -x1 exp(-x0x1), -x0 exp(-x0x1), 20]
  /// with initial condition x(0) = [ 1 ,1 , 1]
  /// this gives a solution 8.332816e-01 3.533462e-02 -4.985493e-01

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
    Kokkos::single(Kokkos::PerTeam(member), [&]() {
      x(0) = 1;
      x(1) = 1;
      x(2) = 1;
    });
    member.team_barrier();
  }

  template<typename MemberType,
           typename RealType1DViewType,
           typename RealType2DViewType>
  KOKKOS_INLINE_FUNCTION void computeJacobian(const MemberType& member,
                                              const RealType1DViewType& x,
                                              const RealType2DViewType& J) const
  {
    Kokkos::single(Kokkos::PerTeam(member), [&]() {
      const real_type x0 = x(0), x1 = x(1), x2 = x(2);
      J(0, 0) = 3;
      J(0, 1) = x2 * ats<real_type>::sin(x1 * x2);
      J(0, 2) = x1 * ats<real_type>::sin(x1 * x2);

      J(1, 0) = 8 * x0;
      J(1, 1) = -1250 * x1;
      J(1, 2) = 2;

      J(2, 0) = -x1 * ats<real_type>::exp(-x0 * x1);
      J(2, 1) = -x0 * ats<real_type>::exp(-x0 * x1);
      J(2, 2) = 20;
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
      f(0) = 3 * x0 - ats<real_type>::cos(x1 * x2) - 1.5;
      f(1) = 4 * x0 * x0 - 625 * x1 * x1 + 2 * x2 - 1;
      f(2) = 20 * x2 + ats<real_type>::exp(-x0 * x1) + 9;
    });
    member.team_barrier();
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
      const ordinal_type xDim = 3;
      /// input with initial values zero
      real_type_2d_view x("x", nBatch, xDim);

      Kokkos::Impl::Timer timer;

      timer.reset();
      using policy_type = Kokkos::TeamPolicy<TChem::exec_space>;
      const ordinal_type level = 0, per_team_extent = 3 + 3 + 9 + 3;
      const ordinal_type per_team_scratch =
        TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);

      policy_type policy(nBatch, Kokkos::AUTO());
      policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

      Kokkos::parallel_for(
        "TChem Newton Example",
        policy,
        KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
          const ordinal_type i = member.league_rank();
          TChem::Example::TestProblemNewton problem;

          /// batch solution
          auto x_at_i = real_type_1d_view(&x(i, 0), 3);

          /// use scratch space
          auto work = TChem::Scratch<real_type_1d_view>(
            member.team_scratch(level), per_team_extent);
          auto wptr = work.data();

          /// necessary views
          auto dx = real_type_1d_view(wptr, 3);
          wptr += 3;
          auto f = real_type_1d_view(wptr, 3);
          wptr += 3;
          auto J = real_type_2d_view(wptr, 3, 3);
          wptr += 9;
          auto w = real_type_1d_view(wptr, 4);
          wptr += 3;

          /// simple setting for newton
          real_type atol = 1e-6, rtol = 1e-5;
          ordinal_type /// m = 3,  /// not used
            max_iter = 100, iter_count = 0, converge = 0;

          /// run the newton iterations
          TChem::Impl::NewtonSolver ::team_invoke(member,
                                                  problem,
                                                  atol,
                                                  rtol,
                                                  max_iter, // input
                                                  x_at_i,   // input/output
                                                  dx,
                                                  f,
                                                  J,
                                                  w, // workspace
                                                  iter_count,
                                                  converge); // output

          Kokkos::single(Kokkos::PerTeam(member), [&]() {
            if (i == 0) {
              if (converge) {
                printf("Test problem converged with a solution %e %e %e\n",
                       x_at_i(0),
                       x_at_i(1),
                       x_at_i(2));
              } else {
                printf(
                  "Test problem failed to converge with iteration count %d\n",
                  iter_count);
              }
            }
          });
        });
      Kokkos::fence(); /// timing purpose
      const real_type t_newton = timer.seconds();

      printf("Time for Newton %e sec\n", t_newton);
    }
  }
  Kokkos::finalize();

  return 0;
}
