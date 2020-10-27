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
#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"

#include "TChem_IgnitionZeroD.hpp"

using ordinal_type = TChem::ordinal_type;
using real_type = TChem::real_type;
using time_advance_type = TChem::time_advance_type;

using real_type_0d_view = TChem::real_type_0d_view;
using real_type_1d_view = TChem::real_type_1d_view;
using real_type_2d_view = TChem::real_type_2d_view;

using time_advance_type_0d_view = TChem::time_advance_type_0d_view;
using time_advance_type_1d_view = TChem::time_advance_type_1d_view;

using real_type_0d_view_host = TChem::real_type_0d_view_host;
using real_type_1d_view_host = TChem::real_type_1d_view_host;
using real_type_2d_view_host = TChem::real_type_2d_view_host;

using time_advance_type_0d_view_host = TChem::time_advance_type_0d_view_host;
using time_advance_type_1d_view_host = TChem::time_advance_type_1d_view_host;

#define TCHEM_EXAMPLE_IGNITIONZEROD_QOI_PRINT

int
main(int argc, char* argv[])
{

  /// default inputs
  std::string prefixPath("data/ignition-zero-d/");
  std::string chemFile(prefixPath + "chem.inp");
  std::string thermFile(prefixPath + "therm.dat");
  std::string inputFile(prefixPath + "input.dat");

  const real_type zero(0);
  real_type tbeg(0), tend(1);
  real_type dtmin(1e-11), dtmax(1e-6);
  real_type rtol_time(1e-8), atol_newton(1e-8), rtol_newton(1e-5);
  int num_time_iterations_per_interval(1e1), max_num_time_iterations(1e3),
    max_num_newton_iterations(100);

  int nBatch(1), team_size(-1), vector_size(-1);
  ;
  bool verbose(true);

  /// parse command line arguments
  TChem::CommandLineParser opts(
    "This example computes reaction rates with a given state vector");
  opts.set_option<std::string>(
    "chemfile", "Chem file name e.g., chem.inp", &chemFile);
  opts.set_option<std::string>(
    "thermfile", "Therm file name e.g., therm.dat", &thermFile);
  opts.set_option<std::string>(
    "inputfile", "Input state file name e.g., input.dat", &inputFile);
  opts.set_option<real_type>("tbeg", "Time begin", &tbeg);
  opts.set_option<real_type>("tend", "Time end", &tend);
  opts.set_option<real_type>("dtmin", "Minimum time step size", &dtmin);
  opts.set_option<real_type>("dtmax", "Maximum time step size", &dtmax);
  opts.set_option<real_type>(
    "atol-newton", "Absolute tolerence used in newton solver", &atol_newton);
  opts.set_option<real_type>(
    "rtol-newton", "Relative tolerence used in newton solver", &rtol_newton);
  opts.set_option<real_type>(
    "tol-time", "Tolerence used for adaptive time stepping", &rtol_time);
  opts.set_option<int>("time-iterations-per-interval",
                       "Number of time iterations per interval to store qoi",
                       &num_time_iterations_per_interval);
  opts.set_option<int>("max-time-iterations",
                       "Maximum number of time iterations",
                       &max_num_time_iterations);
  opts.set_option<int>("max-newton-iterations",
                       "Maximum number of newton iterations",
                       &max_num_newton_iterations);
  opts.set_option<int>(
    "batchsize",
    "Batchsize the same state vector described in statefile is cloned",
    &nBatch);
  opts.set_option<int>("team-size", "User defined team size", &team_size);
  opts.set_option<int>("vector-size", "User defined vector size", &vector_size);
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

    /// construct kmd and use the view for testing
    TChem::KineticModelData kmd(chemFile, thermFile);
    const TChem::KineticModelConstData<TChem::exec_space> kmcd =
      kmd.createConstData<TChem::exec_space>();

    const ordinal_type stateVecDim =
      TChem::Impl::getStateVectorSize(kmcd.nSpec);

    /// input: state vectors: temperature, pressure and concentration
    real_type_2d_view state("StateVector", nBatch, stateVecDim);

    // for (ordinal_type k = 0; k < kmcd.nSpec; k++) {
    //   printf("%k %a\n",k,kmcd.speciesNames(k) );
    // }

    /// create a mirror view to store input from a file
    auto state_host = Kokkos::create_mirror_view(state);

    /// input from a file; this is not necessary as the input is created
    /// by other applications.
    {
      auto state_host_at_i = Kokkos::subview(state_host, 0, Kokkos::ALL());
      TChem::Test::readStateVector(inputFile, kmcd.nSpec, state_host_at_i);
      TChem::Test::cloneView(state_host);
    }

    Kokkos::Impl::Timer timer;
    auto printState = [](const time_advance_type _tadv,
                         const real_type _t,
                         const real_type_1d_view_host _state_at_i) {
#if defined(TCHEM_EXAMPLE_IGNITIONZEROD_QOI_PRINT)
      /// iter, t, dt, rho, pres, temp, Ys ....
      printf("%e %e %e %e %e",
             _t,
             _t - _tadv._tbeg,
             _state_at_i(0),
             _state_at_i(1),
             _state_at_i(2));
      for (ordinal_type k = 3, kend = _state_at_i.extent(0); k < kend; ++k)
        printf(" %e", _state_at_i(k));
      printf("\n");
#endif
    };

    timer.reset();
    Kokkos::deep_copy(state, state_host);
    const real_type t_deepcopy = timer.seconds();

    timer.reset();
    {
      const auto exec_space_instance = TChem::exec_space();
      using policy_type =
        typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;

      /// team policy
      policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());

      const ordinal_type level = 1;
      const ordinal_type per_team_extent =
        TChem::IgnitionZeroD::getWorkSpaceSize(kmcd);
      const ordinal_type per_team_scratch =
        TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
      policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

      { /// time integration
        real_type_1d_view t("time", nBatch);
        Kokkos::deep_copy(t, tbeg);
        real_type_1d_view dt("delta time", nBatch);
        Kokkos::deep_copy(dt, dtmin);

        using problem_type = TChem::Impl::IgnitionZeroD_Problem<decltype(kmcd)>;
        real_type_2d_view tol_time(
          "tol time", problem_type::getNumberOfTimeODEs(kmcd), 2);
        real_type_1d_view tol_newton("tol newton", 2);
        real_type_2d_view fac(
          "fac", nBatch, problem_type::getNumberOfEquations(kmcd));

        /// tune tolerence
        {
          auto tol_time_host = Kokkos::create_mirror_view(tol_time);
          auto tol_newton_host = Kokkos::create_mirror_view(tol_newton);

          const real_type atol_time = 1e-12;
          for (ordinal_type i = 0, iend = tol_time.extent(0); i < iend; ++i) {
            tol_time_host(i, 0) = atol_time;
            tol_time_host(i, 1) = rtol_time;
          }
          tol_newton_host(0) = atol_newton;
          tol_newton_host(1) = rtol_newton;

          Kokkos::deep_copy(tol_time, tol_time_host);
          Kokkos::deep_copy(tol_newton, tol_newton_host);
        }

        time_advance_type tadv_default;
        tadv_default._tbeg = tbeg;
        tadv_default._tend = tend;
        tadv_default._dt = dtmin;
        tadv_default._dtmin = dtmin;
        tadv_default._dtmax = dtmax;
        tadv_default._max_num_newton_iterations = max_num_newton_iterations;
        tadv_default._num_time_iterations_per_interval =
          num_time_iterations_per_interval;

        time_advance_type_1d_view tadv("tadv", nBatch);
        Kokkos::deep_copy(tadv, tadv_default);

        /// host views to print QOI
        const auto tadv_at_i = Kokkos::subview(tadv, 0);
        const auto t_at_i = Kokkos::subview(t, 0);
        const auto state_at_i = Kokkos::subview(state, 0, Kokkos::ALL());

        auto tadv_at_i_host = Kokkos::create_mirror_view(tadv_at_i);
        auto t_at_i_host = Kokkos::create_mirror_view(t_at_i);
        auto state_at_i_host = Kokkos::create_mirror_view(state_at_i);

        ordinal_type iter = 0;
        /// print of store QOI for the first sample
#if defined(TCHEM_EXAMPLE_IGNITIONZEROD_QOI_PRINT)
        {
          /// could use cuda streams
          Kokkos::deep_copy(tadv_at_i_host, tadv_at_i);
          Kokkos::deep_copy(t_at_i_host, t_at_i);
          Kokkos::deep_copy(state_at_i_host, state_at_i);
          printState(tadv_at_i_host(), t_at_i_host(), state_at_i_host);
        }
#endif
        real_type tsum(0);
        for (; iter < max_num_time_iterations && tsum <= tend; ++iter) {
          TChem::IgnitionZeroD::runDeviceBatch(
            policy, tol_newton, tol_time, fac, tadv, state, t, dt, state, kmcd);
          Kokkos::fence();
          /// print of store QOI for the first sample
#if defined(TCHEM_EXAMPLE_IGNITIONZEROD_QOI_PRINT)
          {
            /// could use cuda streams
            Kokkos::deep_copy(tadv_at_i_host, tadv_at_i);
            Kokkos::deep_copy(t_at_i_host, t_at_i);
            Kokkos::deep_copy(state_at_i_host, state_at_i);
            printState(tadv_at_i_host(), t_at_i_host(), state_at_i_host);
          }
#endif
          /// carry over time and dt computed in this step
          tsum = zero;
          Kokkos::parallel_reduce(
            Kokkos::RangePolicy<TChem::exec_space>(0, nBatch),
            KOKKOS_LAMBDA(const ordinal_type& i, real_type& update) {
              tadv(i)._tbeg = t(i);
              tadv(i)._dt = dt(i);
              printf("t %e, dt %e\n", t(i), dt(i));
              printf("tadv t %e, tadv dt %e\n", tadv(i)._tbeg, tadv(i)._dt);
              update += t(i);
            },
            tsum);
          Kokkos::fence();
          tsum /= nBatch;
        }
      }
    }
    Kokkos::fence(); /// timing purpose
    const real_type t_device_batch = timer.seconds();
    printf("Time ignition  %e [sec] %e [sec/sample]\n",
           t_device_batch,
           t_device_batch / real_type(nBatch));
  }
  Kokkos::finalize();

  return 0;
}
