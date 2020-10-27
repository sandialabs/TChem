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
#include "Kokkos_Random.hpp"
#include "TChem_Util.hpp"

#include "TChem_CommandLineParser.hpp"
#include "TChem_Impl_DenseUTV.hpp"

using ordinal_type = TChem::ordinal_type;
using real_type = TChem::real_type;

using real_type_1d_view = TChem::real_type_1d_view;
using real_type_2d_view = TChem::real_type_2d_view;
using real_type_3d_view = TChem::real_type_3d_view;

using real_type_1d_view_host = TChem::real_type_1d_view_host;
using real_type_2d_view_host = TChem::real_type_2d_view_host;
using real_type_3d_view_host = TChem::real_type_3d_view_host;

int
main(int argc, char* argv[])
{

  /// default inputs
  int nBatch(1);
  int matDim = 10;
  int matRank = 2;
  int numRhs = 2;
  bool verbose(true);

  /// parse command line arguments
  TChem::CommandLineParser opts(
    "This example computes reaction rates with a given state vector");
  opts.set_option<int>(
    "batchsize",
    "Batchsize the same state vector described in statefile is cloned",
    &nBatch);
  opts.set_option<int>("mat-dim", "Problem matrix dimension", &matDim);
  opts.set_option<int>("mat-rank", "Problem matrix rank to simulate", &matRank);
  opts.set_option<int>("num-rhs", "# of rhs", &numRhs);
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
      /// problem generation
      real_type_3d_view problem("problem", nBatch, matDim, matDim);
      real_type_3d_view solution("solution", nBatch, matDim, numRhs);
      real_type_3d_view rhs("rhs", nBatch, matDim, numRhs);

      /// single problem generation
      real_type_2d_view_host rand_mat_host(
        "small-rank-vectors", matDim, matRank);
      real_type_2d_view_host single_problem_host(
        "single-problem", matDim, matDim);
      real_type_2d_view_host single_rhs_host("single-rhs", matDim, numRhs);

      Kokkos::Random_XorShift64_Pool<TChem::host_exec_space> random(13245);
      Kokkos::fill_random(rand_mat_host, random, real_type(1));
      Kokkos::fill_random(single_rhs_host, random, real_type(1));

      for (ordinal_type i = 0; i < matDim; ++i)
        for (ordinal_type j = 0; j < matDim; ++j)
          for (ordinal_type k = 0; k < matRank; ++k)
            single_problem_host(i, j) +=
              rand_mat_host(i, k) * rand_mat_host(j, k);

      auto single_problem = Kokkos::create_mirror_view_and_copy(
        typename TChem::exec_space::memory_space(), single_problem_host);
      auto single_rhs = Kokkos::create_mirror_view_and_copy(
        typename TChem::exec_space::memory_space(), single_rhs_host);

      using policy_type = Kokkos::TeamPolicy<TChem::exec_space>;
      {
        /// problems on device
        policy_type policy(nBatch, Kokkos::AUTO());
        Kokkos::parallel_for(
          "TChem Dense UTV Problems",
          policy,
          KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            const ordinal_type idx = member.league_rank();
            auto A =
              Kokkos::subview(problem, idx, Kokkos::ALL(), Kokkos::ALL());
            auto b = Kokkos::subview(rhs, idx, Kokkos::ALL(), Kokkos::ALL());
            Kokkos::parallel_for(
              Kokkos::TeamThreadRange(member, matDim),
              [&](const ordinal_type& i) {
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, matDim),
                                     [&](const ordinal_type& j) {
                                       A(i, j) = single_problem(i, j);
                                     });
                Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(member, numRhs),
                  [&](const ordinal_type& j) { b(i, j) = single_rhs(i, j); });
              });
          });
      }

      {
        const ordinal_type level(0);
        // const ordinal_type m = matDim, n = matDim, nrhs = numRhs;
        // const ordinal_type per_team_extent = m*m + n*n + n + std::min(m,n) +
        // n*nrhs;
        const ordinal_type per_team_extent =
          2 * matDim * matDim + (numRhs + 3) * matDim;
        const ordinal_type per_team_scratch =
          TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);

        policy_type policy(nBatch, Kokkos::AUTO());
        policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
        Kokkos::parallel_for(
          "TChem Dense UTV",
          policy,
          KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            /// use scratch space
            auto s = TChem::Scratch<real_type_1d_view>(
              member.team_scratch(level), per_team_extent);
            real_type_1d_view w(s.data(), s.extent(0));

            const ordinal_type idx = member.league_rank();
            real_type_2d_view A(&problem(idx, 0, 0), matDim, matDim);

            ordinal_type matrix_rank;
            if (numRhs == 1) {
              real_type_1d_view x(&solution(idx, 0, 0), matDim);
              real_type_1d_view b(&rhs(idx, 0, 0), matDim);
              TChem::Impl::DenseUTV::team_factorize_and_solve(
                member, A, x, b, w, matrix_rank);
            } else {
              real_type_2d_view x(&solution(idx, 0, 0), matDim, numRhs);
              real_type_2d_view b(&rhs(idx, 0, 0), matDim, numRhs);
              TChem::Impl::DenseUTV::team_factorize_and_solve(
                member, A, x, b, w, matrix_rank);
            }
            printf("matrix rank = %d\n", matrix_rank);
          });
        Kokkos::fence(); /// timing purpose
      }

      {
        auto A_host = single_problem_host;
        auto b_host = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(),
          Kokkos::subview(rhs, 0, Kokkos::ALL(), Kokkos::ALL()));
        auto x_host = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(),
          Kokkos::subview(solution, 0, Kokkos::ALL(), Kokkos::ALL()));

        real_type_2d_view_host r_host("r", matDim, numRhs),
          t_host("t", matDim, numRhs);

        /// norm(A' ( A x - b)) should be machine zero
        for (ordinal_type i = 0; i < matDim; ++i) {
          for (ordinal_type k = 0; k < numRhs; ++k) {
            real_type temp(0);
            for (ordinal_type j = 0; j < matDim; ++j)
              temp += A_host(i, j) * x_host(j, k);
            t_host(i, k) = temp - b_host(i, k);
          }
        }
        for (ordinal_type i = 0; i < matDim; ++i) {
          for (ordinal_type k = 0; k < numRhs; ++k) {
            real_type temp(0);
            for (ordinal_type j = 0; j < matDim; ++j)
              temp += A_host(j, i) * t_host(j, k);
            r_host(i, k) = temp;
          }
        }
        real_type residual(0), norm(0);
        for (ordinal_type i = 0; i < matDim; ++i) {
          for (ordinal_type j = 0; j < numRhs; ++j) {
            norm += b_host(i, j) * b_host(i, j);
            residual += r_host(i, j) * r_host(i, j);
          }
        }

        // printf("A = \n");
        // for (ordinal_type i=0;i<matDim;++i) {
        //   for (ordinal_type j=0;j<matDim;++j)
        //     printf(" %15.14e ", A_host(i,j));
        //   printf("\n");
        // }
        // printf("x = \n");
        // for (ordinal_type i=0;i<matDim;++i) {
        //   for (ordinal_type j=0;j<numRhs;++j)
        //     printf(" %15.14e ", x_host(i,j));
        //   printf("\n");
        // }
        // printf("b = \n");
        // for (ordinal_type i=0;i<matDim;++i) {
        //   for (ordinal_type j=0;j<numRhs;++j)
        //     printf(" %15.14e ", b_host(i,j));
        //   printf("\n");
        // }

        printf("norm %e residual %e relative residual %e \n",
               norm,
               residual,
               residual / norm);
      }
    }
  }
  Kokkos::finalize();

  return 0;
}
