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
#include "TChem_SimpleSurface.hpp"
#include "TChem_InitialCondSurface.hpp"
#include "TChem_PlugFlowReactor.hpp"


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

#define TCHEM_EXAMPLE_PFR_QOI_PRINT

int
main(int argc, char* argv[])
{
#if defined(TCHEM_ENABLE_TPL_YAML_CPP)
  /// default inputs
  std::string prefixPath("data/plug-flow-reactor/X/");

  real_type Area(0.00053);
  real_type Pcat(0.025977239243415308);

  const real_type zero(0);
  real_type tbeg(0), tend(0.025);
  real_type dtmin(1e-10), dtmax(1e-6);
  real_type atol_time(1e-12), rtol_time(1e-4), atol_newton(1e-12), rtol_newton(1e-6);
  int num_time_iterations_per_interval(1e1), max_num_time_iterations(4e3),
    max_num_newton_iterations(20), jacobian_interval(1);

  int nBatch(1), team_size(-1), vector_size(-1);
  bool verbose(true);

  bool transient_initial_condition(false);
  bool initial_condition(true);

  std::string chemFile("chem.inp");
  std::string thermFile("therm.dat");
  std::string chemSurfFile("chemSurf.inp");
  std::string thermSurfFile("thermSurf.dat");
  std::string inputFile("sample.dat");
  std::string inputFileSurf( "inputSurf.dat");
  std::string inputFilevelocity( "inputVelocity.dat");
  std::string outputFile("PFRSolution.dat");
  std::string inputFileParamModifiers("ParameterModifiers.dat");

  bool use_prefixPath(true);
  /// parse command line arguments
  TChem::CommandLineParser opts(
    "This example computes Temperature, density, mass fraction and site "
    "fraction for a plug flow reactor");

  opts.set_option<bool>(
      "use-prefix-path", "If true, input file are at the prefix path", &use_prefixPath);

  opts.set_option<std::string>(
    "inputs-path", "prefixPath e.g.,inputs/", &prefixPath);

  opts.set_option<std::string>
  ("chemfile", "Chem file name e.g., chem.inp",
  &chemFile);

  opts.set_option<std::string>
  ("thermfile", "Therm file name e.g., therm.dat", &thermFile);

  opts.set_option<std::string>
  ("surf-chemfile","Chem file name e.g., chemSurf.inp",
   &chemSurfFile);

  opts.set_option<std::string>
  ("surf-thermfile", "Therm file name e.g.,thermSurf.dat",
  &thermSurfFile);

  opts.set_option<std::string>
  ("samplefile", "Input state file name e.g., input.dat", &inputFile);

  opts.set_option<std::string>
  ("surf-inputfile", "Input state file name e.g., inputSurfGas.dat", &inputFileSurf);

  opts.set_option<std::string>
  ("velocity-inputfile", "Input state file name e.g., inputVelocity.dat", &inputFilevelocity);

  opts.set_option<std::string>("outputfile",
  "Output file name e.g., PFRSolution.dat", &outputFile);
  opts.set_option<real_type>("reactor-area", "Cross-sectional Area [m2]", &Area);
  opts.set_option<real_type>("catalytic-perimeter", "Chemically active perimeter [m],", &Pcat);
  opts.set_option<std::string>
  ("input-file-param-modifiers", "Input state file name e.g.,input.dat", &inputFileParamModifiers);

  opts.set_option<real_type>("zbeg", "Position begin", &tbeg);
  opts.set_option<real_type>("zend", "Position end", &tend);
  opts.set_option<real_type>("dzmin", "Minimum dz step size", &dtmin);
  opts.set_option<real_type>("dzmax", "Maximum dz step size", &dtmax);
  opts.set_option<real_type>(
    "atol-newton", "Absolute tolerance used in newton solver", &atol_newton);
  opts.set_option<real_type>(
    "rtol-newton", "Relative tolerance used in newton solver", &rtol_newton);
  opts.set_option<real_type>(
    "tol-z", "Tolerance used for adaptive z stepping", &rtol_time);
  opts.set_option<real_type>(
    "atol-z", "Absolute tolerence used for adaptive time stepping", &atol_time);
  opts.set_option<int>("time-iterations-per-interval",
                       "Number of time iterations per interval to store qoi",
                       &num_time_iterations_per_interval);
  opts.set_option<int>("max-z-iterations",
                       "Maximum number of z iterations",
                       &max_num_time_iterations);
  opts.set_option<int>("max-newton-iterations",
                       "Maximum number of newton iterations",
                       &max_num_newton_iterations);
  opts.set_option<int>(
    "batchsize",
    "Batchsize the same state vector described in statefile is cloned",
    &nBatch);
  opts.set_option<bool>(
    "verbose", "If true, printout the first Jacobian values", &verbose);
  opts.set_option<int>("jacobian-interval", "Jacobians are evaluated in this interval during Newton solve", &jacobian_interval);
  opts.set_option<int>("team-size", "User defined team size", &team_size);
  opts.set_option<int>("vector-size", "User defined vector size", &vector_size);
  opts.set_option<bool>(
    "transient-initial-condition", "If true, use a transient solver to obtain initial condition of the constraint", &transient_initial_condition);
    opts.set_option<bool>(
      "initial-condition", "If true, use a newton solver to obtain initial condition of the constraint", &initial_condition);


  const bool r_parse = opts.parse(argc, argv);
  if (r_parse)
    return 0; // print help return

    // if one wants to all the input files in one directory,
    //and do not want give all names
    if ( use_prefixPath ){
      chemFile      = prefixPath + "chem.inp";
      thermFile     = prefixPath + "therm.dat";
      chemSurfFile  = prefixPath + "chemSurf.inp";
      thermSurfFile = prefixPath + "thermSurf.dat";
      inputFile     = prefixPath + "sample.dat";
      inputFileSurf = prefixPath + "inputSurf.dat";
      inputFilevelocity = prefixPath + "inputVelocity.dat";
      inputFileParamModifiers = prefixPath + "ParameterModifiers.dat";
      printf("Using a prefix path %s \n",prefixPath.c_str() );
    }

  Kokkos::initialize(argc, argv);
  {

    const bool detail = false;

    TChem::exec_space::print_configuration(std::cout, detail);
    TChem::host_exec_space::print_configuration(std::cout, detail);

    using device_type      = typename Tines::UseThisDevice<exec_space>::type;
    /// construct kmd and use the view for testing

    TChem::KineticModelData kmd(
      chemFile, thermFile, chemSurfFile, thermSurfFile);
    const auto kmcd =
      TChem::createGasKineticModelConstData<device_type>(kmd); // data struc with gas phase info
    const auto kmcdSurf =
      TChem::createSurfaceKineticModelConstData<device_type>(kmd); // data struc with

    const ordinal_type stateVecDim =
      TChem::Impl::getStateVectorSize(kmcd.nSpec);

    real_type_2d_view_host state_host;
    real_type_2d_view_host siteFraction_host;
    real_type_1d_view_host velocity_host;

    const auto speciesNamesHost = Kokkos::create_mirror_view(kmcd.speciesNames);
    Kokkos::deep_copy(speciesNamesHost, kmcd.speciesNames);

    // get names of species on host
    const auto SurfSpeciesNamesHost =
      Kokkos::create_mirror_view(kmcdSurf.speciesNames);
    Kokkos::deep_copy(SurfSpeciesNamesHost, kmcdSurf.speciesNames);

    {
      // get names of species on host

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

      TChem::Test::readSurfaceSample(inputFileSurf,
                                     SurfSpeciesNamesHost,
                                     kmcdSurf.nSpec,
                                     siteFraction_host,
                                     nBatch);

      // read velocity (velocity)
      velocity_host = real_type_1d_view_host("Velocity host", nBatch);
      TChem::Test::read1DVector(inputFilevelocity, nBatch, velocity_host);

      if (state_host.extent(0) != siteFraction_host.extent(0) ||
          state_host.extent(0) != velocity_host.extent(0))
        std::logic_error("Error: number of sample is not valid");
    }

    real_type_2d_view siteFraction("SiteFraction", nBatch, kmcdSurf.nSpec);
    real_type_2d_view state("StateVector ", nBatch, stateVecDim);
    real_type_1d_view velocity("Velocity", nBatch);

    const YAML::Node inputFile = YAML::LoadFile(inputFileParamModifiers);



    /// different sample would have a different kinetic model
    //gas phase
    auto kmds = kmd.clone(nBatch);
    if (inputFile["Gas"])
    {
      printf("reading gas phase ...\n");
      constexpr ordinal_type gas(0);
      const auto DesignOfExperimentGas = inputFile["Gas"]["DesignOfExperiment"];
      TChem::modifyArrheniusForwardParameter(kmds, gas, DesignOfExperimentGas);
    }

    auto kmcds = TChem::createGasKineticModelConstData<device_type>(kmds);

    if (inputFile["Surface"])
    {
      printf("reading surface phase ...\n");
      constexpr ordinal_type surface(1);
      const auto DesignOfExperimentSurface = inputFile["Surface"]["DesignOfExperiment"];
      TChem::modifyArrheniusForwardParameter(kmds, surface, DesignOfExperimentSurface);
    }
    auto kmcdSurfs = TChem::createSurfaceKineticModelConstData<device_type>(kmds);


    if (inputFile["Gas"])
    {
      TChem::Test::printParametersModelVariation("Gas", kmcd, kmcds, inputFile);
    }

    if (inputFile["Surface"])
    {
      TChem::Test::printParametersModelVariation("Surface", kmcdSurf, kmcdSurfs, inputFile);
    }

    Kokkos::Impl::Timer timer;

    FILE* fout = fopen(outputFile.c_str(), "w");

    auto writeState =
      [](const ordinal_type iter,
         const real_type_1d_view_host _t,
         const real_type_1d_view_host _dt,
         const real_type_2d_view_host _state,
         const real_type_2d_view_host _siteFraction,
         const real_type_1d_view_host _vel,
         FILE* fout) { // iteration, time, dt, density, pressure, temperature,
                       // mass fraction, site fraction, velocity
        for (size_t sp = 0; sp < _state.extent(0); sp++) {
          fprintf(fout, "%d \t %15.10e \t  %15.10e \t ", iter, _t(sp), _dt(sp));
          //
          for (ordinal_type k = 0, kend = _state.extent(1); k < kend; ++k)
            fprintf(fout, "%15.10e \t", _state(sp, k));
          //
          for (ordinal_type k = 0, kend = _siteFraction.extent(1); k < kend;
               ++k)
            fprintf(fout, "%15.10e \t", _siteFraction(sp, k));
          fprintf(fout, "%15.10e \t", _vel(sp));

          fprintf(fout, "\n");
        }

      };


    auto printState = [](const time_advance_type _tadv,
                         const real_type _t,
                         const real_type_1d_view_host _state_at_i,
                         const real_type_1d_view_host _siteFraction_at_i) {
#if defined(TCHEM_EXAMPLE_PFR_QOI_PRINT)
      /// iter, t, dt, rho, pres, temp, Ys ....
      printf(" Gas \n");
      printf(" z %e dz   %e density %e  pres %e temp %e",
             _t,
             _t - _tadv._tbeg,
             _state_at_i(0),
             _state_at_i(1),
             _state_at_i(2));
      printf("\n");
      printf("Mass fraction \n");
      for (ordinal_type k = 3, kend = _state_at_i.extent(0); k < kend; ++k)
        printf(" %e", _state_at_i(k));
      printf("\n");
      printf("Site Fraction \n");
      for (ordinal_type k = 0, kend = _siteFraction_at_i.extent(0); k < kend;
           ++k)
        printf(" %e", _siteFraction_at_i(k));
      printf("\n");
#endif
    };

    timer.reset();
    Kokkos::deep_copy(state, state_host);
    Kokkos::deep_copy(siteFraction, siteFraction_host);
    Kokkos::deep_copy(velocity, velocity_host);

    const real_type t_deepcopy = timer.seconds();
    // Run a batch reactor simulation with constant gas conditions.

    const auto exec_space_instance = TChem::exec_space();
    using policy_type =
      typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;

    if (transient_initial_condition) {

      policy_type policy_surf(exec_space_instance, nBatch, Kokkos::AUTO());

      using problem_type_surf =
        Impl::SimpleSurface_Problem<real_type,
                                    device_type>;

      const ordinal_type level = 1;
      const ordinal_type per_team_extent =
        TChem::SimpleSurface::getWorkSpaceSize(kmcd, kmcdSurf);
      ordinal_type per_team_scratch =
        TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
      policy_surf.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

        real_type_2d_view fac_surf("fac simple", nBatch, problem_type_surf::getNumberOfTimeODEs(kmcdSurf));

      { /// time integration
        real_type_1d_view t_surf("time", nBatch);
        Kokkos::deep_copy(t_surf, 0);
        real_type_1d_view dt_surf("delta time", nBatch);
        Kokkos::deep_copy(dt_surf, dtmin);

        time_advance_type tadv_default_surf;
        tadv_default_surf._tbeg = 0;
        tadv_default_surf._tend = 1;
        tadv_default_surf._dt = dtmin;
        tadv_default_surf._dtmin = dtmin;
        tadv_default_surf._dtmax = dtmax;
        tadv_default_surf._max_num_newton_iterations = max_num_newton_iterations;
        tadv_default_surf._num_time_iterations_per_interval = 1000;
        tadv_default_surf._jacobian_interval = jacobian_interval;

        time_advance_type_1d_view tadv_surf("tadv simple surface", nBatch);
        Kokkos::deep_copy(tadv_surf, tadv_default_surf);

        real_type_2d_view tol_time_surf(
          "tol time simple surface", problem_type_surf::getNumberOfTimeODEs(kmcdSurf), 2);
        real_type_1d_view tol_newton_surf("tol newton simple surface", 2);

        /// tune tolerence
        {
          auto tol_time_host_surf = Kokkos::create_mirror_view(tol_time_surf);
          auto tol_newton_host_surf = Kokkos::create_mirror_view(tol_newton_surf);


          for (ordinal_type i = 0, iend = tol_time_surf.extent(0); i < iend; ++i) {
            tol_time_host_surf(i, 0) = atol_time;
            tol_time_host_surf(i, 1) = rtol_time;
          }
          tol_newton_host_surf(0) = atol_newton;
          tol_newton_host_surf(1) = rtol_newton;

          Kokkos::deep_copy(tol_time_surf, tol_time_host_surf);
          Kokkos::deep_copy(tol_newton_surf, tol_newton_host_surf);
        }

        ordinal_type iter = 0;
        const ordinal_type  max_num_time_iterations_surface(1);

        real_type tsum(0);
        for (; iter < max_num_time_iterations_surface && tsum <= tend*0.999; ++iter) {
          TChem::SimpleSurface::runDeviceBatch(policy_surf,
                                               tol_newton_surf,
                                               tol_time_surf,
                                               tadv_surf,
                                               state,
                                               siteFraction,
                                               t_surf,
                                               dt_surf,
                                               siteFraction,
                                               fac_surf,
                                               kmcds,
                                               kmcdSurfs);

          /// carry over time and dt computed in this step
          tsum = zero;
          Kokkos::parallel_reduce(
            Kokkos::RangePolicy<TChem::exec_space>(0, nBatch),
            KOKKOS_LAMBDA(const ordinal_type& i, real_type& update) {
              tadv_surf(i)._tbeg = t_surf(i);
              tadv_surf(i)._dt = dt_surf(i);
              update += t_surf(i);
            },
            tsum);
          Kokkos::fence();
          tsum /= nBatch;
        }

      }

    printf("Done with transient initial Condition for DAE system\n");

    }
    //Run a newton solver to check that constraint is satified.

    if (initial_condition){
      real_type_2d_view facSurf("facSurf", nBatch, kmcdSurf.nSpec);

      real_type_1d_view tol_newton_initial_condition("tol newton initial conditions", 2);
      auto tol_newton_host_initial_condition = Kokkos::create_mirror_view(tol_newton_initial_condition);

      tol_newton_host_initial_condition(0) = 1e-12/*atol_newton*/;
      tol_newton_host_initial_condition(1) = 1e-8/*rtol_newton*/;
      Kokkos::deep_copy(tol_newton_initial_condition, tol_newton_host_initial_condition);
      const real_type max_num_newton_iterations_initial_condition(1000);

      /// team policy
      policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());

      const ordinal_type level = 1;
      const ordinal_type per_team_extent =
        TChem::InitialCondSurface::getWorkSpaceSize(kmcd, kmcdSurf);
      const ordinal_type per_team_scratch =
        TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
      policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

      // solve initial condition for PFR solution
      TChem::InitialCondSurface::runDeviceBatch(policy,
                                                tol_newton_initial_condition,
                                                max_num_newton_iterations_initial_condition,
                                                state,
                                                siteFraction, // input
                                                siteFraction, // output
                                                facSurf,
                                                kmcds,
                                                kmcdSurfs);

      printf("Done with initial Condition for DAE system\n");
    }

    timer.reset();
    {

      {
        PlugFlowReactorData pfrd;
        pfrd.Area = Area; // m2
        pfrd.Pcat = Pcat; //

        real_type_1d_view t("position", nBatch);
        Kokkos::deep_copy(t, tbeg);
        real_type_1d_view dt("delta z", nBatch);
        Kokkos::deep_copy(dt, dtmin);

        real_type_1d_view_host t_host;
        real_type_1d_view_host dt_host;

        t_host = real_type_1d_view_host("position host", nBatch);
        dt_host = real_type_1d_view_host("dz host", nBatch);

        using problem_type =
          TChem::Impl::PlugFlowReactor_Problem<real_type,device_type>;
        real_type_2d_view tol_time(
          "tol z", problem_type::getNumberOfTimeODEs(kmcd), 2);
        real_type_1d_view tol_newton("tol z", 2);

        real_type_2d_view fac(
          "fac", nBatch, problem_type::getNumberOfEquations(kmcd, kmcdSurf));

        /// tune tolerence
        {
          auto tol_time_host = Kokkos::create_mirror_view(tol_time);
          auto tol_newton_host = Kokkos::create_mirror_view(tol_newton);
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
        const auto siteFraction_at_i =
          Kokkos::subview(siteFraction, 0, Kokkos::ALL());
        const auto velocity_at_i = Kokkos::subview(velocity, 0);

        auto tadv_at_i_host = Kokkos::create_mirror_view(tadv_at_i);
        auto t_at_i_host = Kokkos::create_mirror_view(t_at_i);
        auto state_at_i_host = Kokkos::create_mirror_view(state_at_i);
        auto siteFraction_at_i_host =
          Kokkos::create_mirror_view(siteFraction_at_i);
        auto velocity_at_i_host = Kokkos::create_mirror_view(velocity_at_i);

        ordinal_type iter = 0;
        /// print of store QOI for the first sample
#if defined(TCHEM_EXAMPLE_PFR_QOI_PRINT)
        {
          /// could use cuda streams
          Kokkos::deep_copy(tadv_at_i_host, tadv_at_i);
          Kokkos::deep_copy(t_at_i_host, t_at_i);
          Kokkos::deep_copy(state_at_i_host, state_at_i);
          Kokkos::deep_copy(siteFraction_at_i_host, siteFraction_at_i);
          printState(tadv_at_i_host(),
                     t_at_i_host(),
                     state_at_i_host,
                     siteFraction_at_i_host);
        }
#endif

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
          //
        for (ordinal_type k = 0; k < kmcdSurf.nSpec; k++)
          fprintf(fout, "%s \t", &SurfSpeciesNamesHost(k, 0));

        fprintf(fout, "%s \t", "velocity[m/s]");

        fprintf(fout, "\n");
          // save initial condition
        Kokkos::deep_copy(siteFraction_host, siteFraction);

        writeState(-1,
                   t_host,
                   dt_host,
                   state_host,
                   siteFraction_host,
                   velocity_host,
                  fout);


        using PFRSolver = TChem::PlugFlowReactor;
        /// team policy
        policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());

        const ordinal_type level = 1;
        const ordinal_type per_team_extent =
          TChem::PlugFlowReactor::getWorkSpaceSize(kmcd, kmcdSurf);
        const ordinal_type per_team_scratch =
          TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
        policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

        real_type tsum(0);
        for (; iter < max_num_time_iterations && tsum <= tend*0.9999; ++iter) {

          PFRSolver::runDeviceBatch(policy,
                                                 tol_newton,
                                                 tol_time,
                                                 fac,
                                                 tadv,
                                                 // input
                                                 state,
                                                 siteFraction,
                                                 velocity,
                                                 // output
                                                 t,
                                                 dt,
                                                 state,
                                                 siteFraction,
                                                 velocity,
                                                 // kinetic info
                                                 kmcds,
                                                 kmcdSurfs,
                                                 // pfrd_info,
                                                 Area,
                                                 Pcat);
          //

          /// print of store QOI for the first sample
#if defined(TCHEM_EXAMPLE_PFR_QOI_PRINT)
          {
            /// could use cuda streams
            Kokkos::deep_copy(tadv_at_i_host, tadv_at_i);
            Kokkos::deep_copy(t_at_i_host, t_at_i);
            Kokkos::deep_copy(state_at_i_host, state_at_i);
            Kokkos::deep_copy(siteFraction_at_i_host, siteFraction_at_i);
            Kokkos::deep_copy(velocity_at_i_host, velocity_at_i);
            printState(tadv_at_i_host(),
                       t_at_i_host(),
                       state_at_i_host,
                       siteFraction_at_i_host);
          }
#endif

            // time, sample, state
            // save time and dt

            Kokkos::deep_copy(dt_host, dt);
            Kokkos::deep_copy(t_host, t);
            Kokkos::deep_copy(state_host, state);
            Kokkos::deep_copy(siteFraction_host, siteFraction);
            Kokkos::deep_copy(velocity_host, velocity);

            writeState(iter,
                       t_host,
                       dt_host,
                       state_host,
                       siteFraction_host,
                       velocity_host,
                       fout);

          /// carry over time and dt computed in this step
          tsum = zero;
          Kokkos::parallel_reduce(
            Kokkos::RangePolicy<TChem::exec_space>(0, nBatch),
            KOKKOS_LAMBDA(const ordinal_type& i, real_type& update) {
              tadv(i)._tbeg = t(i);
              tadv(i)._dt = dt(i);
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
    printf("Time   %e [sec] %e [sec/sample]\n",
           t_device_batch,
           t_device_batch / real_type(nBatch));

    fclose(fout);
  }
  Kokkos::finalize();
  #else
   printf("This example requires Yaml ...\n" );
  #endif

  return 0;
}
