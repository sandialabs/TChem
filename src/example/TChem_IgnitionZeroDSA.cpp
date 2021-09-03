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
#include "TChem_ConstantVolumeIgnitionReactor.hpp"

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

// #define TCHEM_EXAMPLE_IGNITIONZEROD_QOI_PRINT

int
main(int argc, char* argv[])
{

  /// default inputs
  std::string prefixPath("data/ignition-zero-d/CO/");

  const real_type zero(0);
  real_type tbeg(0), tend(1);
  real_type dtmin(1e-8), dtmax(1e-1);
  real_type rtol_time(1e-4), atol_newton(1e-10), rtol_newton(1e-6);
  int num_time_iterations_per_interval(1e1), max_num_time_iterations(1e3),
    max_num_newton_iterations(100), jacobian_interval(1);
  int output_frequency(1);

  real_type T_threshold(1500);

  int nBatch(1), team_size(-1), vector_size(-1);
  ;
  bool verbose(true);
  bool OnlyComputeIgnDelayTime(false);
  bool run_ignition_zero_d(true);

  std::string chemFile("chem.inp");
  std::string thermFile("therm.dat");
  std::string inputFile("sample.dat");
  std::string outputFile("IgnSolution.dat");
  std::string ignition_delay_time_file("IgnitionDelayTime.dat");
  std::string ignition_delay_time_w_threshold_temperature_file("IgnitionDelayTimeTthreshold.dat");

  bool use_prefixPath(true);

  /// parse command line arguments
  TChem::CommandLineParser opts(
    "This example computes the solution of an ignition problem");
  opts.set_option<std::string>(
    "inputsPath", "path to input files e.g., data/inputs", &prefixPath);
  opts.set_option<bool>(
      "use_prefixPath", "If true, input file are at the prefix path", &use_prefixPath);

  opts.set_option<std::string>
  ("chemfile", "Chem file name e.g., chem.inp",&chemFile);
  opts.set_option<std::string>
  ("thermfile", "Therm file namee.g., therm.dat", &thermFile);
  opts.set_option<std::string>
  ("samplefile", "Input state file name e.g.,input.dat", &inputFile);
  opts.set_option<real_type>("tbeg", "Time begin", &tbeg);
  opts.set_option<real_type>("tend", "Time end", &tend);
  opts.set_option<real_type>("dtmin", "Minimum time step size", &dtmin);
  opts.set_option<real_type>("dtmax", "Maximum time step size", &dtmax);
  opts.set_option<real_type>(
    "atol-newton", "Absolute tolerance used in newton solver", &atol_newton);
  opts.set_option<real_type>(
    "rtol-newton", "Relative tolerance used in newton solver", &rtol_newton);
  opts.set_option<bool>(
    "run-ignition-zero-d",
    "if true code runs ignition zero d reactor; else code runs constant volume ignition reactor", &run_ignition_zero_d);
  opts.set_option<std::string>("outputfile",
    "Output file name e.g., IgnSolution.dat", &outputFile);
  opts.set_option<std::string>("ignition-delay-time-file",
    "Output of ignition delay time second using second-derivative method e.g., IgnitionDelayTime.dat",
     &ignition_delay_time_file);
  opts.set_option<std::string>("ignition-delay-time-w-threshold-temperature-file",
    "Output of ignition delay time second using threshold-temperature method  e.g., IgnitionDelayTimeTthreshold.dat",
    &ignition_delay_time_w_threshold_temperature_file);
  opts.set_option<real_type>(
    "tol-time", "Tolerance used for adaptive time stepping", &rtol_time);
  opts.set_option<int>("time-iterations-per-interval",
                       "Number of time iterations per interval to store qoi",
                       &num_time_iterations_per_interval);
  opts.set_option<int>("max-time-iterations",
                       "Maximum number of time iterations",
                       &max_num_time_iterations);
  opts.set_option<int>("jacobian-interval",
                       "Jacobians are evaluated once in this interval",
                       &jacobian_interval);
  opts.set_option<int>("max-newton-iterations",
                       "Maximum number of newton iterations",
                       &max_num_newton_iterations);
  opts.set_option<int>(
    "output_frequency", "save data at this iterations", &output_frequency);
  // opts.set_option<int>("batchsize", "Batchsize the same state vector
  // described in statefile is cloned", &nBatch);
  opts.set_option<bool>(
    "verbose", "If true, printout the first Jacobian values", &verbose);
  opts.set_option<real_type>(
    "T_threshold", "Temp threshold in ignition delay time", &T_threshold);

  //
  opts.set_option<bool>(
    "OnlyComputeIgnDelayTime",
    "If true, simulation will end when Temperature is equal to T_threshold ",
    &OnlyComputeIgnDelayTime);

  opts.set_option<int>("team-size", "User defined team size", &team_size);
  opts.set_option<int>("vector-size", "User defined vector size", &vector_size);

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse)
    return 0; // print help return

    // if ones wants to all the input files in one directory,
    //and do not want give all names
    if ( use_prefixPath ){
      chemFile      = prefixPath + "chem.inp";
      thermFile     = prefixPath + "therm.dat";
      inputFile     = prefixPath + "sample.dat";

      printf("Using a prefix path %s \n",prefixPath.c_str() );
    }

  Kokkos::initialize(argc, argv);
  {
      printf("----------------------------------------------------- \n");
      printf("---------------------WARNING------------------------- \n");
    if (run_ignition_zero_d){
      printf("   TChem is running Ignition Zero D Reactor \n");
    } else {
      printf("   TChem is running Constant Volume Ignition Reactor \n");

    }
      printf("----------------------------------------------------- \n");
      printf("----------------------------------------------------- \n");

    const bool detail = false;

    TChem::exec_space::print_configuration(std::cout, detail);
    TChem::host_exec_space::print_configuration(std::cout, detail);
    using device_type      = typename Tines::UseThisDevice<exec_space>::type;


    /// construct kmd and use the view for testing
    TChem::KineticModelData kmd(chemFile, thermFile);
    const TChem::KineticModelConstData<device_type> kmcd =
      kmd.createConstData<device_type>();

    const ordinal_type stateVecDim =
      TChem::Impl::getStateVectorSize(kmcd.nSpec);

    printf("Number of Species %d \n", kmcd.nSpec);
    printf("Number of Reactions %d \n", kmcd.nReac);

    if (OnlyComputeIgnDelayTime) {
      printf("This simulation will end when tempearature is equal to "
             "T_threshold (K) %e\n",
             T_threshold);
    }

    FILE* fout = fopen(outputFile.c_str(), "w");

    /// input from a file; this is not necessary as the input is created
    /// by other applications.
    // real_type_2d_view state;
    real_type_2d_view_host state_host;
    const auto speciesNamesHost = Kokkos::create_mirror_view(kmcd.speciesNames);
    Kokkos::deep_copy(speciesNamesHost, kmcd.speciesNames);
    {
      // get species molecular weigths
      const auto SpeciesMolecularWeights =
        Kokkos::create_mirror_view(kmcd.sMass);
      Kokkos::deep_copy(SpeciesMolecularWeights, kmcd.sMass);

      TChem::Test::readSample(inputFile,
                              speciesNamesHost,
                              SpeciesMolecularWeights,
                              kmcd.nSpec,
                              stateVecDim,
                              state_host,
                              nBatch);
    }
    real_type_2d_view state("StateVector Devices", nBatch, stateVecDim);

    // #endif
    // Temperature at iter and iter-1 for each sample, row: sample, colum iter
    // and iter-1
    real_type_2d_view TempIgn("TemperatureComputeIgn", nBatch, 2);
    // Time row: sample col: iter and iter-1
    real_type_2d_view TimeIgn("TimeComputeIgn", nBatch, 2);
    // ingition delay times for each sample
    real_type_1d_view IgnDelayTimes("IngitionDelayTimes", nBatch);
    real_type_1d_view IgnDelayTimesT("IngitionDelayTimesTempthreshold", nBatch);

    // make ingition delay negative
    Kokkos::parallel_for(
      Kokkos::RangePolicy<TChem::exec_space>(0, nBatch),
      KOKKOS_LAMBDA(const ordinal_type& i) {
        IgnDelayTimes(i) = -1;
        IgnDelayTimesT(i) = -1;
      });

    Kokkos::Impl::Timer timer;

    timer.reset();
    Kokkos::deep_copy(state, state_host);
    const real_type t_deepcopy = timer.seconds();

#if defined(TCHEM_EXAMPLE_IGNITIONZEROD_QOI_PRINT)

    {
      for (ordinal_type i = 0; i < nBatch; i++) {
        printf("Host::Initial condition sample No %d\n", i);
        const auto state_at_i_host =
          Kokkos::subview(state_host, i, Kokkos::ALL());
        for (ordinal_type k = 0, kend = state_at_i_host.extent(0); k < kend;
             ++k)
          printf(" %e", state_at_i_host(k));
        printf("\n");
      }
    }

    Kokkos::parallel_for(
      Kokkos::RangePolicy<TChem::exec_space>(0, nBatch),
      KOKKOS_LAMBDA(const ordinal_type& i) {
        printf("Devices::Initial condition sample No %d\n", i);
        auto state_at_i = Kokkos::subview(state, i, Kokkos::ALL());
        for (ordinal_type k = 0, kend = state_at_i.extent(0); k < kend; ++k)
          printf(" %e", state_at_i(k));
        printf("\n");
      });

#endif

    //

    auto writeState = [](const ordinal_type iter,
                         const real_type_1d_view_host _t,
                         const real_type_1d_view_host _dt,
                         const real_type_2d_view_host _state_at_i,
                         FILE* fout) { // sample, time, density, pressure,
                                       // temperature, mass fraction
      for (size_t sp = 0; sp < _state_at_i.extent(0); sp++) {
        fprintf(fout, "%d \t %15.10e \t  %15.10e \t ", iter, _t(sp), _dt(sp));
        //
        for (ordinal_type k = 0, kend = _state_at_i.extent(1); k < kend; ++k)
          fprintf(fout, "%15.10e \t", _state_at_i(sp, k));

        fprintf(fout, "\n");
      }

    };

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
    {
      const auto exec_space_instance = TChem::exec_space();

      using policy_type =
        typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;

      /// team policy
      policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());

      if (team_size > 0 && vector_size > 0) {
        policy = policy_type(exec_space_instance, nBatch, team_size, vector_size);
      }

      const ordinal_type level = 1;
      ordinal_type per_team_extent(0);

      if (run_ignition_zero_d){
        per_team_extent = TChem::IgnitionZeroD::getWorkSpaceSize(kmcd);
      } else {
        per_team_extent = TChem::ConstantVolumeIgnitionReactor::getWorkSpaceSize(kmcd);
      }

      const ordinal_type per_team_scratch =
        TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
      policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

      { /// time integration
        real_type_1d_view t("time", nBatch);
        Kokkos::deep_copy(t, tbeg);
        real_type_1d_view dt("delta time", nBatch);
        Kokkos::deep_copy(dt, dtmax);

        real_type_1d_view_host t_host;
        real_type_1d_view_host dt_host;

        if (output_frequency > 0) {
          t_host = real_type_1d_view_host("time host", nBatch);
          dt_host = real_type_1d_view_host("dt host", nBatch);
        }

        ordinal_type number_of_equations(0);

        if (run_ignition_zero_d){
          using problem_type = TChem::Impl::IgnitionZeroD_Problem<real_type, device_type>;
          number_of_equations = problem_type::getNumberOfTimeODEs(kmcd);
        }else {
          using problem_type = TChem::Impl::ConstantVolumeIgnitionReactor_Problem<real_type, device_type>;
          number_of_equations = problem_type::getNumberOfTimeODEs(kmcd);
        }

        real_type_2d_view tol_time(
          "tol time", number_of_equations, 2);
        real_type_1d_view tol_newton("tol newton", 2);

        real_type_2d_view fac(
          "fac", nBatch, number_of_equations);

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
        tadv_default._jacobian_interval = jacobian_interval;

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

        //
        if (output_frequency > 0) {
          const ordinal_type indx = iter / output_frequency;
          printf("save at iteration %d indx %d\n", iter, indx);
          // time, sample, state
          // save time and dt

          Kokkos::deep_copy(dt_host, dt);
          Kokkos::deep_copy(t_host, t);

          fprintf(fout, "%s \t %s \t %s \t ", "iter", "t", "dt");
          fprintf(fout,
                  "%s \t %s \t %s \t",
                  "Density[kg/m3]",
                  "Pressure[Pascal]",
                  "Temperature[K]");

          for (ordinal_type k = 0; k < kmcd.nSpec; k++)
            fprintf(fout, "%s \t", &speciesNamesHost(k, 0));
          fprintf(fout, "\n");
          // save initial condition
          writeState(-1, t_host, dt_host, state_host, fout);
        }

        real_type tsum(0);
        for (; iter < max_num_time_iterations && tsum <= tend*0.9999; ++iter) {

          if (run_ignition_zero_d){
          TChem::IgnitionZeroD::runDeviceBatch(
            policy, tol_newton, tol_time, fac, tadv, state, t, dt, state, kmcd);
          } else {
          ConstantVolumeIgnitionReactor::runDeviceBatch(
            policy, tol_newton, tol_time, fac, tadv, state, t, dt, state, kmcd);
          }

          /// print of store QOI for the first sample
          /// Ignition delay time Temperature threshold

          Kokkos::parallel_for(
            Kokkos::RangePolicy<TChem::exec_space>(0, nBatch),
            KOKKOS_LAMBDA(const ordinal_type& i) {
              if (IgnDelayTimesT(i) < 0 && state(i, 2) >= T_threshold) {
                IgnDelayTimesT(i) = t(i);

                if (OnlyComputeIgnDelayTime) {
                  t(i) = tend;
                }
                //  // end this sample
              }
            });

          ///* Ignition delay time - second derivative */
          Kokkos::parallel_for(
            Kokkos::RangePolicy<TChem::exec_space>(0, nBatch),
            KOKKOS_LAMBDA(const ordinal_type& i) {
              if (IgnDelayTimes(i) <= 0) {
                const real_type temp_at_i = state(i, 2);
                // dTdt at i-3/2
                // at first iteration TempIgn and TimeIgn are zero
                // It takes two iterations to fill out an value in TempIgn(i,0)
                const real_type dTdt_at_in32 =
                  TempIgn(i, 0) == 0 ? 0
                                     : (TempIgn(i, 1) - TempIgn(i, 0)) /
                                         (TimeIgn(i, 1) - TimeIgn(i, 0));

                // dTdt at i-1/2
                const real_type dTdt_at_in12 =
                  TempIgn(i, 0) == 0
                    ? 0
                    : (temp_at_i - TempIgn(i, 1)) / (t(i) - TimeIgn(i, 1));
                const real_type dT2dt2 = dTdt_at_in12 - dTdt_at_in32;

                if (dT2dt2 < 0 && temp_at_i > real_type(1300.)) {
                  IgnDelayTimes(i) = TimeIgn(i, 1);
                }

                // save iter-1 in iter-2
                TimeIgn(i, 0) = TimeIgn(i, 1);
                TempIgn(i, 0) = TempIgn(i, 1);
                // update time and temperature
                TimeIgn(i, 1) = t(i);
                TempIgn(i, 1) = temp_at_i; // temperature
              }
            });

#if defined(TCHEM_EXAMPLE_IGNITIONZEROD_QOI_PRINT)
          {
            /// could use cuda streams
            Kokkos::deep_copy(tadv_at_i_host, tadv_at_i);
            Kokkos::deep_copy(t_at_i_host, t_at_i);
            Kokkos::deep_copy(state_at_i_host, state_at_i);
            printState(tadv_at_i_host(), t_at_i_host(), state_at_i_host);
          }
#endif

          if (iter % output_frequency == 0 && output_frequency > 0) {
            const ordinal_type indx = iter / output_frequency;
            printf("save at iteration %d indx %d\n", iter, indx);
            // time, sample, state
            // save time and dt

            Kokkos::deep_copy(dt_host, dt);
            Kokkos::deep_copy(t_host, t);
            Kokkos::deep_copy(state_host, state);

            writeState(iter, t_host, dt_host, state_host, fout);

            // Kokkos::parallel_for(
            //   Kokkos::RangePolicy<TChem::exec_space>(0, nBatch),
            //   KOKKOS_LAMBDA(const ordinal_type &i) {
            //     output(indx, i, 0) = iter;
            //     output(indx, i, 1) = t(i);
            //     output(indx, i, 2) = dt(i);
            //     for (ordinal_type k = 0; k < stateVecDim; k++)
            //       output(indx, i, k + 3) = state(i, k);
            //   });
          }

          /// carry over time and dt computed in this step
          tsum = zero;
          Kokkos::parallel_reduce(
            Kokkos::RangePolicy<TChem::exec_space>(0, nBatch),
            KOKKOS_LAMBDA(const ordinal_type& i, real_type& update) {
              tadv(i)._tbeg = t(i);
              tadv(i)._dt = dt(i);
              // printf("t %e, dt %e\n", t(i), dt(i));
              // printf("tadv t %e, tadv dt %e\n", tadv(i)._tbeg, tadv(i)._dt );
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

#if defined(TCHEM_EXAMPLE_IGNITIONZEROD_QOI_PRINT)
    Kokkos::parallel_for(
      Kokkos::RangePolicy<TChem::exec_space>(0, nBatch),
      KOKKOS_LAMBDA(const ordinal_type& i) {
        printf("Devices:: Solution sample No %d\n", i);
        auto state_at_i = Kokkos::subview(state, i, Kokkos::ALL());
        for (ordinal_type k = 0, kend = state_at_i.extent(0); k < kend; ++k)
          printf(" %e", state_at_i(k));
        printf("\n");
      });
#endif


    auto IgnDelayTimes_host = Kokkos::create_mirror_view(IgnDelayTimes);
    Kokkos::deep_copy(IgnDelayTimes_host, IgnDelayTimes);
    TChem::Test::write1DVector(ignition_delay_time_file, IgnDelayTimes_host);

    auto IgnDelayTimesT_host = Kokkos::create_mirror_view(IgnDelayTimesT);
    Kokkos::deep_copy(IgnDelayTimesT_host, IgnDelayTimesT);
    TChem::Test::write1DVector(ignition_delay_time_w_threshold_temperature_file,
                               IgnDelayTimesT_host);

    fclose(fout);
  }
  Kokkos::finalize();

  return 0;
}
