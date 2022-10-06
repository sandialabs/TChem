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
#include "TChem.hpp"
#include "TChem_CommandLineParser.hpp"
#include "TChem_Impl_IgnitionZeroD.hpp"

// #define TCHEM_ENABLE_SAVE_SOLUTION

int main(int argc, char *argv[]) {

  /// default inputs
  std::string prefixPath("data/ignition-zero-d/gri3.0/");
  const real_type zero(0);
  real_type tbeg(0), tend(1);
  real_type dtmin(1e-8), dtmax(1e-1);
  real_type rtol_time(1e-4), atol_newton(1e-10), rtol_newton(1e-6);
  int num_time_iterations_per_interval(1e1), max_num_time_iterations(1e3), max_num_newton_iterations(100),
      jacobian_interval(1);
  real_type atol_time(1e-12);

  bool verbose(true);

  std::string chemFile(prefixPath + "chem.inp");
  std::string thermFile(prefixPath + "therm.dat");
  std::string inputFile(prefixPath + "input.dat");
  std::string outputFile("IgnSolution.dat");

  bool use_prefixPath(false);

  bool use_cvode(false);

  /// parse command line arguments
  TChem::CommandLineParser opts("This example computes the solution of a gas ignition 0D - problem");
  opts.set_option<std::string>("inputs-path", "path to input files e.g., data/inputs", &prefixPath);
  opts.set_option<bool>("use-prefix-path", "If true, input file are at the prefix path", &use_prefixPath);
  opts.set_option<std::string>("chemfile", "Chem file name e.g., chem.inp", &chemFile);
  opts.set_option<std::string>("thermfile", "Therm file namee.g., therm.dat", &thermFile);
  opts.set_option<std::string>("samplefile", "Input state file name e.g.,input.dat", &inputFile);
  opts.set_option<real_type>("tbeg", "Time begin", &tbeg);
  opts.set_option<real_type>("tend", "Time end", &tend);
  opts.set_option<real_type>("dtmin", "Minimum time step size", &dtmin);
  opts.set_option<real_type>("dtmax", "Maximum time step size", &dtmax);
  opts.set_option<real_type>("atol-newton", "Absolute tolerance used in newton solver", &atol_newton);
  opts.set_option<real_type>("rtol-newton", "Relative tolerance used in newton solver", &rtol_newton);

#if defined(TINES_ENABLE_TPL_SUNDIALS)
  opts.set_option<bool>("use-cvode", "if true code runs ignition zero d reactor with cvode; otherwise, it uses TrBDF2",
                        &use_cvode);
#endif

  opts.set_option<std::string>("outputfile", "Output file name e.g., IgnSolution.dat", &outputFile);
  opts.set_option<real_type>("tol-time", "Tolerance used for adaptive time stepping", &rtol_time);
  opts.set_option<real_type>("atol-time", "Absolute tolerance used for adaptive time stepping", &atol_time);
  opts.set_option<int>("time-iterations-per-interval", "Number of time iterations per interval to store qoi",
                       &num_time_iterations_per_interval);
  opts.set_option<int>("max-time-iterations", "Maximum number of time iterations", &max_num_time_iterations);
  opts.set_option<int>("jacobian-interval", "Jacobians are evaluated once in this interval", &jacobian_interval);
  opts.set_option<int>("max-newton-iterations", "Maximum number of newton iterations", &max_num_newton_iterations);
  // described in statefile is cloned", &nBatch);
  opts.set_option<bool>("verbose", "If true, printout the first Jacobian values", &verbose);

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse)
    return 0; // print help return

  if (use_prefixPath) {
    chemFile = prefixPath + "chem.inp";
    thermFile = prefixPath + "therm.dat";
    inputFile = prefixPath + "input.dat";

    printf("Using a prefix path %s \n", prefixPath.c_str());
  }

  Kokkos::initialize(argc, argv);
  {
    const bool detail = false;

    TChem::exec_space::print_configuration(std::cout, detail);
    TChem::host_exec_space::print_configuration(std::cout, detail);
    const auto exec_space_instance = TChem::exec_space();

    /// construct kmd and use the view for testing
    TChem::KineticModelData kmd(chemFile, thermFile);
    const auto kmcd = TChem::createGasKineticModelConstData<interf_host_device_type>(kmd);

    const ordinal_type stateVecDim = TChem::Impl::getStateVectorSize(kmcd.nSpec);

    real_type_2d_view_host state_host;
    const auto speciesNamesHost = kmcd.speciesNames;
    int nBatch(0);
    {
      // get species molecular weigths
      const auto SpeciesMolecularWeights = kmcd.sMass;
      TChem::Test::readSample(inputFile, speciesNamesHost, SpeciesMolecularWeights, kmcd.nSpec, stateVecDim, state_host,
                              nBatch);
    }

    real_type_1d_view_host state = Kokkos::subview(state_host, 0, Kokkos::ALL());

    // real_type_1d_view_host state("state vector", stateVecDim);
    // TChem::Test::readStateVector(inputFile, kmcd.nSpec, state);

    real_type_1d_view_host work("work", TChem::IgnitionZeroD::getWorkSpaceSize(kmcd));

    real_type_0d_view_host t("time");
    real_type_0d_view_host dt("delta time");

    using problem_type = TChem::Impl::IgnitionZeroD_Problem<real_type, interf_host_device_type>;
    const ordinal_type number_of_equations = problem_type::getNumberOfTimeODEs(kmcd);

    real_type_2d_view_host tol_time("tol time", number_of_equations, 2);
    real_type_1d_view_host tol_newton("tol newton", 2);

    real_type_1d_view_host fac("fac", number_of_equations);

    /// tune tolerence
    {
      // const real_type atol_time = 1e-12;
      for (ordinal_type i = 0, iend = tol_time.extent(0); i < iend; ++i) {
        tol_time(i, 0) = atol_time;
        tol_time(i, 1) = rtol_time;
      }
      tol_newton(0) = atol_newton;
      tol_newton(1) = rtol_newton;
    }

    time_advance_type tadv;
    tadv._tbeg = tbeg;
    tadv._tend = tend;
    tadv._dt = dtmin;
    tadv._dtmin = dtmin;
    tadv._dtmax = dtmax;
    tadv._max_num_newton_iterations = max_num_newton_iterations;
    tadv._num_time_iterations_per_interval = num_time_iterations_per_interval;
    tadv._jacobian_interval = jacobian_interval;

#if defined(TCHEM_ENABLE_SAVE_SOLUTION)
    FILE *fout = fopen(outputFile.c_str(), "w");

    auto writeSolution = [](const ordinal_type iter, const real_type _t, const real_type _dt,
                            const real_type_1d_view_host &_state,
                            FILE *fout) { // sample, time, density, pressure,
                                          // temperature, mass fraction
      fprintf(fout, "%d \t %15.10e \t  %15.10e \t ", iter, _t, _dt);
      for (ordinal_type k = 0, kend = _state.extent(0); k < kend; ++k)
        fprintf(fout, "%15.10e \t", _state(k));
      fprintf(fout, "\n");

    };
#endif

    auto writeEndSolutionAndWallTime = [](Impl::StateVector<real_type_1d_view_host> &_sv, const real_type _tend,
                                          const real_type _wall_time, const real_type _atol, const real_type _rtol,
                                          const ordinal_type _number_time_steps,
                                          FILE *fout) { // sample, time, density, pressure,
                                                        // temperature, mass fraction
      // temperature
      fprintf(fout, "%15.10e \t", _sv.Temperature());
      const auto Ys = _sv.MassFractions();
      // masss fractions
      for (ordinal_type k = 0, kend = Ys.extent(0); k < kend; ++k)
        fprintf(fout, "%15.10e \t", Ys(k));
      // final time, wall time
      fprintf(fout, "%15.10e \t  %15.10e \t ", _tend, _wall_time);
      // abs tolerance, rel tolerance
      fprintf(fout, "%15.10e \t  %15.10e \t ", _atol, _rtol);
      // Successful steps (total number of time steps), Successful steps/Jac Calls  (we do not save these quantities)
      fprintf(fout, "%d \t  %d \t ", _number_time_steps, 0);
      fprintf(fout, "\n");

    };

    ordinal_type iter = 0;
    // save header
#if defined(TCHEM_ENABLE_SAVE_SOLUTION)
    fprintf(fout, "%s \t %s \t %s \t ", "iter", "t", "dt");
    fprintf(fout, "%s \t %s \t %s \t", "Density[kg/m3]", "Pressure[Pascal]", "Temperature[K]");
    for (ordinal_type k = 0; k < kmcd.nSpec; k++)
      fprintf(fout, "%s \t", &kmcd.speciesNames(k, 0));
    fprintf(fout, "\n");

    // save intial condition
    writeSolution(-1, 0, 0, state, fout);
#endif

    const auto member = Tines::HostSerialTeamMember();

    Impl::StateVector<real_type_1d_view_host> sv(kmcd.nSpec, state);
    Impl::StateVector<real_type_1d_view_host> sv_out(kmcd.nSpec, state);

    TCHEM_CHECK_ERROR(!sv.isValid(), "Error: input state vector is not valid");
    TCHEM_CHECK_ERROR(!sv_out.isValid(), "Error: input state vector is not valid");
    const auto Ys = sv.MassFractions();

    const real_type_0d_view_host temperature_out(sv_out.TemperaturePtr());
    const real_type_0d_view_host pressure_out(sv_out.PressurePtr());
    const real_type_1d_view_host Ys_out = sv_out.MassFractions();
    const real_type_0d_view_host density_out(sv_out.DensityPtr());

    const ordinal_type m = kmcd.nSpec + 1;
    auto wptr = work.data();
    const real_type_1d_view_host vals(wptr, m);
    wptr += m;
    const real_type_1d_view_host ww(wptr, work.extent(0) - (wptr - work.data()));

    //
    Kokkos::Timer timer;

    if (!use_cvode) {
      printf("   TChem is running Ignition Zero D Reactor with TrBDF2 \n");

      timer.reset();
      for (; iter < max_num_time_iterations && t() <= tend * 0.9999; ++iter) {

        const auto temperature = sv.Temperature();
        const auto pressure = sv.Pressure();

        if (t() < tadv._tend) {
          /// we can only guarantee vals is contiguous array. we basically assume
          /// that a state vector can be arbitrary ordered.

          /// m is nSpec + 1
          Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                               [&](const ordinal_type &i) { vals(i) = i == 0 ? temperature : Ys(i - 1); });

          using ignition_zeroD_type = Impl::IgnitionZeroD<real_type, interf_host_device_type>;

          ignition_zeroD_type::team_invoke(member, tadv._jacobian_interval, tadv._max_num_newton_iterations,
                                           tadv._num_time_iterations_per_interval, tol_newton, tol_time, fac, tadv._dt,
                                           tadv._dtmin, tadv._dtmax, tadv._tbeg, tadv._tend, pressure, vals, t, dt,
                                           pressure_out, vals, ww, kmcd);

          Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m), [&](const ordinal_type &i) {
            if (i == 0) {
              temperature_out() = vals(0);
            } else {
              Ys_out(i - 1) = vals(i);
            }
          });
          //
          density_out() = Impl::RhoMixMs<real_type, interf_host_device_type>::team_invoke(member, temperature_out(),
                                                                                          pressure_out(), Ys_out, kmcd);
        } // if t_out
        /// carry over time and dt computed in this step
        tadv._tbeg = t();
        tadv._dt = dt();

#if defined(TCHEM_ENABLE_SAVE_SOLUTION)
        writeSolution(iter, t(), dt(), state, fout);
#endif

      }                // end time iter
      Kokkos::fence(); /// timing purpose
      const real_type t_integration = timer.seconds();

      const ordinal_type total_num_iter = iter * tadv._num_time_iterations_per_interval;
      FILE *fout_wall = fopen("trbdf2_wall_times.dat", "w");
      writeEndSolutionAndWallTime(sv, t(), t_integration, atol_time, rtol_time, total_num_iter, fout_wall);
      fclose(fout_wall);
    }
#if defined(TCHEM_ENABLE_SAVE_SOLUTION)
    fclose(fout);
#endif

#if defined(TINES_ENABLE_TPL_SUNDIALS)
    using time_integrator_cvode_type = Tines::TimeIntegratorCVODE<real_type, interf_host_device_type>;

    if (use_cvode) {

      printf("   TChem is running Ignition Zero D Reactor CVODE\n");

#if defined(TCHEM_ENABLE_SAVE_SOLUTION)
      FILE *fout_cvode = fopen(outputFile.c_str(), "w");

      // save header
      fprintf(fout_cvode, "%s \t %s \t %s \t ", "iter", "t", "dt");
      fprintf(fout_cvode, "%s \t %s \t %s \t", "Density[kg/m3]", "Pressure[Pascal]", "Temperature[K]");
      for (ordinal_type k = 0; k < kmcd.nSpec; k++)
        fprintf(fout_cvode, "%s \t", &kmcd.speciesNames(k, 0));
      fprintf(fout_cvode, "\n");

      writeSolution(-1, 0, 0, state, fout_cvode);
#endif

      Tines::value_type_1d_view<time_integrator_cvode_type, interf_host_device_type> cvodes("cvodes", 1);
      cvodes(0).create(kmcd.nSpec + 1);

      auto cvode = cvodes(0);

      const ordinal_type m = kmcd.nSpec + 1;
      auto vals = cvode.getStateVector();

      /// problem setup
      const ordinal_type problem_workspace_size = problem_type::getWorkSpaceSize(kmcd);

      problem_type problem;
      problem._kmcd = kmcd;

      /// problem workspace
      auto wptr = work.data();
      auto pw = real_type_1d_view_host(wptr, problem_workspace_size);
      wptr += problem_workspace_size;

      /// error check
      const ordinal_type workspace_used(wptr - work.data()), workspace_extent(work.extent(0));
      if (workspace_used > workspace_extent) {
        Kokkos::abort("Error Ignition ZeroD Sacado : workspace used is larger than it is provided\n");
      }

      /// time integrator workspace
      auto tw = real_type_1d_view_host(wptr, workspace_extent - workspace_used);

      /// initialize problem

      problem._work = pw;   // problem workspace array
      problem._kmcd = kmcd; // kinetic model
      problem._fac = fac;
      problem._work_cvode = tw; // time integrator workspace
      // constant pressure; for other type of reactors, pressure will need to be updated inside of the time loop
      const auto pressure = sv.Pressure();
      problem._p = pressure; // pressure
      ordinal_type iter = 0;
      timer.reset();
      for (; iter < max_num_time_iterations && t() <= tend * 0.9999; ++iter) {
        const auto temperature = sv.Temperature();
        if (t() < tadv._tend) {
          {
            const real_type dt_in = tadv._dt, dt_min = tadv._dtmin, dt_max = tadv._dtmax;
            const real_type t_beg = tadv._tbeg;
            /// m is nSpec + 1
            vals(0) = temperature;
            for (ordinal_type i = 1; i < m; ++i)
              vals(i) = Ys(i - 1);
            real_type t_out = t(), dt_out = 0;
            cvode.initialize(t_out, dt_in, dt_min, dt_max, tol_time(0, 0), tol_time(0, 1),
                             TChem::Impl::ProblemIgnitionZeroD_ComputeFunctionCVODE,
                             TChem::Impl::ProblemIgnitionZeroD_ComputeJacobianCVODE);

            cvode.setProblem(problem);
            for (ordinal_type iter_internal = 0;
                 iter_internal <= tadv._num_time_iterations_per_interval && t_out <= tadv._tend; ++iter_internal) {
              const real_type t_prev = t_out;
              cvode.advance(tadv._tend, t_out, 1);
              dt_out = t_out - t_prev;
            }

            t() = t_out;
            dt() = dt_out;

            temperature_out() = vals(0);
            for (ordinal_type i = 1; i < m; ++i)
              Ys_out(i - 1) = vals(i);

            density_out() = Impl::RhoMixMs<real_type, interf_host_device_type>::team_invoke(
                member, temperature_out(), pressure_out(), Ys_out, kmcd);
          }

#if defined(TCHEM_ENABLE_SAVE_SOLUTION)
          writeSolution(iter, t(), dt(), state, fout_cvode);
#endif
          tadv._tbeg = t();
          tadv._dt = dt();
        }
      }
      Kokkos::fence(); /// timing purpose
      const real_type t_integration = timer.seconds();

      const ordinal_type total_num_iter = iter * tadv._num_time_iterations_per_interval;
      FILE *fout_wall = fopen("cvode_wall_times.dat", "w");
      writeEndSolutionAndWallTime(sv, t(), t_integration, atol_time, rtol_time, total_num_iter, fout_wall);
      fclose(fout_wall);

#if defined(TCHEM_ENABLE_SAVE_SOLUTION)
      fclose(fout_cvode);
#endif
    }

#endif
  }
  Kokkos::finalize();

  return 0;
}
