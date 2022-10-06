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
#include "TChem_EnthalpyMass.hpp"
#include "TChem_TransientContStirredTankReactor.hpp"
#include "TChem_IsothermalTransientContStirredTankReactor.hpp"

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


#define TCHEM_EXAMPLE_SimpleSurface_QOI_PRINT


// #if defined(TCHEM_ENABLE_PROBLEM_DAE_CSTR)
#include "TChem_SimpleSurface.hpp"
#include "TChem_InitialCondSurface.hpp"
// #endif

int
main(int argc, char* argv[])
{

  /// default inputs
  std::string prefixPath("runs/T-CSTR/CH4_PT_Quinceno2006/inputs/");

  std::string chemFile(prefixPath+"chemgri30.inp");
  std::string thermFile(prefixPath+"thermgri30.dat");
  std::string chemSurfFile(prefixPath+"chemSurf.inp");
  std::string thermSurfFile(prefixPath+"thermSurf.dat");
  std::string inputFile(prefixPath+"sample_phi1.dat");
  std::string inputFileSurf(prefixPath+"inputSurf.dat");
  std::string outputFile("CSTRSolution.dat");

  real_type mdotIn(1e-2);
  real_type Vol(1.347e-1);
  real_type Acat (1.347e-2);

  const real_type zero(0);
  real_type tbeg(0), tend(3);
  real_type dtmin(1e-20), dtmax(1e-2);
  real_type atol_time(1e-12),rtol_time(1e-4), atol_newton(1e-18), rtol_newton(1e-8);
  int num_time_iterations_per_interval(1e1), max_num_time_iterations(10),
    max_num_newton_iterations(100), jacobian_interval(1);

  int nBatch(1), team_size(-1), vector_size(-1);
  bool verbose(true);

  bool use_prefixPath(false);
  bool isothermal(false);
  bool saveInitialCondition(true);

  bool transient_initial_condition(false);
  bool initial_condition(false);
  int number_of_algebraic_constraints(0);
  int poisoning_species_idx(-1);
  bool useYaml(false);

  TChem::CommandLineParser opts(
    "This example computes temperature, mass fraction, and site "
    "fraction for a Transient continuous stirred tank reactor");
  opts.set_option<std::string>(
    "inputs-path", "prefixPath e.g.,inputs/", &prefixPath);
  opts.set_option<bool>(
      "use-prefix-path", "If true, input file are at the prefix path", &use_prefixPath);
  opts.set_option<real_type>("catalytic-area", "Catalytic area [m2]", &Acat);
  opts.set_option<real_type>("reactor-volume", "Reactor Volumen [m3]", &Vol);
  opts.set_option<real_type>("inlet-mass-flow", "Inlet mass flow rate [kg/s]", &mdotIn);
  opts.set_option<bool>("isothermal", "if True, reaction is isotermic", &isothermal);
  opts.set_option<bool>("save-initial-condition", "if True, solution containts"
                        "initial condition", &saveInitialCondition);
  opts.set_option<bool>(
    "transient-initial-condition", "If true, use a transient solver "
    "to obtain initial condition of surface species", &transient_initial_condition);
  opts.set_option<bool>(
      "initial-condition", "If true, use a newton solver to obtain initial"
      "condition of surface species", &initial_condition);
  opts.set_option<std::string>
  ("chemfile", "Chem file name of gas phase e.g., chem.inp",
  &chemFile);
  opts.set_option<std::string>
  ("thermfile", "Therm file name of gas phase  e.g., therm.dat", &thermFile);

  opts.set_option<std::string>
  ("surf-chemfile","Chem file name of surface phase e.g., chemSurf.inp",
   &chemSurfFile);

  opts.set_option<std::string>
  ("surf-thermfile", "Therm file name of surface phase e.g.,thermSurf.dat",
  &thermSurfFile);

  opts.set_option<std::string>
  ("samplefile", "Input state file name of gas phase e.g., input.dat", &inputFile);

  opts.set_option<std::string>
  ("surf-inputfile", "Input state file name of surface e.g., inputSurfGas.dat", &inputFileSurf);

  opts.set_option<real_type>("tbeg", "Time begin", &tbeg);
  opts.set_option<real_type>("tend", "Time end", &tend);
  opts.set_option<real_type>("dtmin", "Minimum time step size", &dtmin);
  opts.set_option<real_type>("dtmax", "Maximum time step size", &dtmax);
  opts.set_option<real_type>(
    "atol-newton", "Absolute tolerance used in newton solver", &atol_newton);
  opts.set_option<real_type>(
    "rtol-newton", "Relative tolerance used in newton solver", &rtol_newton);
  opts.set_option<real_type>(
    "tol-time", "Tolerence used for adaptive time stepping", &rtol_time);
  opts.set_option<real_type>(
    "atol-time", "Absolute tolerance used for adaptive time stepping", &atol_time);
  opts.set_option<int>("time-iterations-per-interval",
                       "Number of time iterations per interval to store qoi",
                       &num_time_iterations_per_interval);
  opts.set_option<int>("max-time-iterations",
                       "Maximum number of time iterations",
                       &max_num_time_iterations);
  opts.set_option<int>("max-newton-iterations",
                       "Maximum number of newton iterations",
                       &max_num_newton_iterations);
  opts.set_option<int>("number-of-algebraic-constraints",
                       "Number of algebraic constraints",
                       &number_of_algebraic_constraints);
  opts.set_option<int>("jacobian-interval", "Jacobians are evaluated in "
  "this interval during Newton solve", &jacobian_interval);
  opts.set_option<bool>(
    "verbose", "If true, printout the first Jacobian values", &verbose);
  opts.set_option<int>("team-size", "User defined team size", &team_size);
  opts.set_option<int>("vector-size", "User defined vector size", &vector_size);
  opts.set_option<int>("index-poisoning-species",
                       "catalysis deactivation, index for species",
                       &poisoning_species_idx);
  opts.set_option<bool>(
    "use-yaml", "If true, use yaml to parse input file", &useYaml);
  opts.set_option<std::string>("outputfile",
  "Output file name e.g., CSTRSolution.dat", &outputFile);


  const bool r_parse = opts.parse(argc, argv);
  if (r_parse)
    return 0; // print help return


  // if ones wants to all the input files in one directory,
  //and do not want give all names
  if ( use_prefixPath ){
    chemFile      = prefixPath + "chem.inp";
    thermFile     = prefixPath + "therm.dat";
    chemSurfFile  = prefixPath + "chemSurf.inp";
    thermSurfFile = prefixPath + "thermSurf.dat";
    inputFile     = prefixPath + "sample.dat";
    inputFileSurf = prefixPath + "inputSurf.dat";
    printf("Using a prefix path %s \n",prefixPath.c_str() );
  }



  Kokkos::initialize(argc, argv);
  {
    const bool detail = false;

    TChem::exec_space::print_configuration(std::cout, detail);
    TChem::host_exec_space::print_configuration(std::cout, detail);

    if (verbose) {
      printf("Inlet mass flow %e kg/s \n", mdotIn );
      printf("Catalytic Area %e m2 \n",Acat);
      printf("Vol %e m3 \n",Vol );
    }

    using device_type      = typename Tines::UseThisDevice<exec_space>::type;

    const auto exec_space_instance = TChem::exec_space();
    using policy_type =
      typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;

    /// construct kmd and use the view for testing
    TChem::KineticModelData kmd;
    if (useYaml) {
      kmd = TChem::KineticModelData(chemFile, true);
    } else {
      kmd = TChem::KineticModelData(chemFile, thermFile, chemSurfFile, thermSurfFile);
    }
    
    const auto kmcd = TChem::createGasKineticModelConstData<device_type>(kmd);
    const auto kmcdSurf = TChem::createSurfaceKineticModelConstData<device_type>(kmd);

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


      if (state_host.extent(0) != siteFraction_host.extent(0) )
        std::logic_error("Error: number of sample is not valid");
    }

    real_type_2d_view siteFraction("SiteFraction", nBatch, kmcdSurf.nSpec);
    real_type_2d_view state("StateVector ", nBatch, stateVecDim);

    Kokkos::Timer timer;

    FILE* fout = fopen(outputFile.c_str(), "w");
    FILE* fout_mass_flow = fopen(("outlet_mass_flow_"+outputFile).c_str(), "w");
    fprintf(fout_mass_flow, "{\n");

    auto writeState =
      [](const ordinal_type iter,
         const real_type_1d_view_host _t,
         const real_type_1d_view_host _dt,
         const real_type_2d_view_host _state,
         const real_type_2d_view_host _siteFraction,
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

          fprintf(fout, "\n");
        }

      };

    auto writeOutletMassFlowRate =
      [] (const ordinal_type iter,
          const real_type_1d_view_host _t,
          const real_type_1d_view_host _dt,
          const real_type_1d_view_host _outlet_mass_flow,
          FILE* fout)
          {
            for (size_t sp = 0; sp < _t.extent(0); sp++)
              fprintf(fout, " \"nBacth_%d_iter_%d\":[%15.10e,  %15.10e, %15.10e],\n", sp, iter, _t(sp), _dt(sp), _outlet_mass_flow(sp));
          };



    auto printState = [](const time_advance_type _tadv,
                         const real_type _t,
                         const real_type_1d_view_host _state_at_i,
                         const real_type_1d_view_host _siteFraction_at_i) {
#if defined(TCHEM_EXAMPLE_SimpleSurface_QOI_PRINT)
      /// iter, t, dt, rho, pres, temp, Ys ....
      printf(" Gas \n");
      printf(" t %e dt %e density %e  pres %e temp %e",
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

// #if defined(TCHEM_ENABLE_PROBLEM_DAE_CSTR)

    if (transient_initial_condition) {

      printf("Running transient initial condition \n" );

      policy_type policy_surf(exec_space_instance, nBatch, Kokkos::AUTO());

      const ordinal_type level = 1;
      const ordinal_type per_team_extent =
        TChem::SimpleSurface::getWorkSpaceSize(kmcd, kmcdSurf);
      ordinal_type per_team_scratch =
        TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
      policy_surf.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

        real_type_2d_view fac_surf("fac simple", nBatch, kmcdSurf.nSpec);

      { /// time integration
        real_type_1d_view t_surf("time", nBatch);
        Kokkos::deep_copy(t_surf, 0);
        real_type_1d_view dt_surf("delta time", nBatch);
        Kokkos::deep_copy(dt_surf, 1e-20);

        time_advance_type tadv_default_surf;
        tadv_default_surf._tbeg = 0;
        tadv_default_surf._tend = 1;
        tadv_default_surf._dt = 1e-20;
        tadv_default_surf._dtmin = 1e-20;
        tadv_default_surf._dtmax = 1e-3;
        tadv_default_surf._max_num_newton_iterations = 20;
        tadv_default_surf._num_time_iterations_per_interval = 2000;
        tadv_default_surf._jacobian_interval = jacobian_interval;

        time_advance_type_1d_view tadv_surf("tadv simple surface", nBatch);
        Kokkos::deep_copy(tadv_surf, tadv_default_surf);

        real_type_2d_view tol_time_surf(
          "tol time simple surface", kmcdSurf.nSpec, 2);
        real_type_1d_view tol_newton_surf("tol newton simple surface", 2);

        /// tune tolerence
        {
          auto tol_time_host_surf = Kokkos::create_mirror_view(tol_time_surf);
          auto tol_newton_host_surf = Kokkos::create_mirror_view(tol_newton_surf);

          const real_type atol_time_surf = 1e-12;
          const real_type rtol_time_surf = 1e-8;

          const real_type atol_newton_surf = 1e-18;
          const real_type rtol_newton_surf = 1e-8;

          for (ordinal_type i = 0, iend = tol_time_surf.extent(0); i < iend; ++i) {
            tol_time_host_surf(i, 0) = atol_time_surf;
            tol_time_host_surf(i, 1) = rtol_time_surf;
          }
          tol_newton_host_surf(0) = atol_newton_surf;
          tol_newton_host_surf(1) = rtol_newton_surf;

          Kokkos::deep_copy(tol_time_surf, tol_time_host_surf);
          Kokkos::deep_copy(tol_newton_surf, tol_newton_host_surf);
        }

        ordinal_type iter = 0;
        const ordinal_type  max_num_time_iterations_surface(1);

        real_type tsum(0);
        for (; iter < max_num_time_iterations_surface && tsum <= tend; ++iter) {
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
                                               kmcd,
                                               kmcdSurf);

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


    }
    //Run a newton solver to check that constraint is satified.
    //
    // if (initial_condition){
    //   real_type_2d_view facSurf("facSurf", nBatch, kmcdSurf.nSpec);
    //
    //   /// team policy
    //   policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());
    //
    //   const ordinal_type level = 1;
    //   const ordinal_type per_team_extent =
    //     TChem::InitialCondSurface::getWorkSpaceSize(kmcd, kmcdSurf);
    //   const ordinal_type per_team_scratch =
    //     TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
    //   policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
    //
    //   // solve initial condition for PFR solution
    //   TChem::InitialCondSurface::runDeviceBatch(policy,
    //                                             state,
    //                                             siteFraction, // input
    //                                             siteFraction, // output
    //                                             facSurf,
    //                                             kmcd,
    //                                             kmcdSurf);
    //
    //   printf("Done with initial Condition for DAE system\n");
    // }

// #endif
    const real_type t_deepcopy = timer.seconds();

    timer.reset();
    {

      {

        //setting up cstr reactor
        printf("Setting up CSTR reactor\n");
        TransientContStirredTankReactorData<device_type> cstr;
        cstr.mdotIn = mdotIn; // inlet mass flow kg/s
        cstr.Vol    = Vol; // volumen of reactor m3
        cstr.Acat   = Acat; // Catalytic area m2: chemical active area
        cstr.pressure = state_host(0, 1);
        cstr.isothermal = 1;
        if (isothermal) cstr.isothermal = 0; // 0 constant temperature
        // cstr.temperature = state_host(0, 2);
        if (number_of_algebraic_constraints > kmcdSurf.nSpec) {
          number_of_algebraic_constraints = kmcdSurf.nSpec;

          printf("------------------------------------------------------------------------- \n");
          printf("-----------------------------------WARNING------------------------------- \n");
          printf(" Number of algebraic constraints is bigger than number of surface species \n");
          printf(" Setting number of algebraic constrains equal to number of surface species %d \n", kmcdSurf.nSpec);
          printf("------------------------------------------------------------------------- \n");
          printf("------------------------------------------------------------------------- \n");

        }

        cstr.number_of_algebraic_constraints = number_of_algebraic_constraints;

        cstr.Yi = real_type_1d_view("Mass fraction at inlet", kmcd.nSpec);
        printf("Reactor residence time [s] %e\n", state_host(0, 0)*cstr.Vol/cstr.mdotIn);
        cstr.poisoning_species_idx=poisoning_species_idx;

        // work batch = 1
        Kokkos::parallel_for(
          Kokkos::RangePolicy<TChem::exec_space>(0, nBatch),
          KOKKOS_LAMBDA(const ordinal_type& i) {
            //mass fraction
            for (ordinal_type k = 0; k < kmcd.nSpec; k++) {
              cstr.Yi(k) = state(i,k+3);
            }
        });



        real_type_1d_view t("time", nBatch);
        Kokkos::deep_copy(t, tbeg);
        real_type_1d_view dt("delta time", nBatch);
        Kokkos::deep_copy(dt, dtmin);

        real_type_1d_view_host t_host;
        real_type_1d_view_host dt_host;

        {
          t_host = real_type_1d_view_host("time host", nBatch);
          dt_host = real_type_1d_view_host("dt host", nBatch);
        }

        using problem_type =
          TChem::Impl::TransientContStirredTankReactor_Problem<real_type,device_type>;
        real_type_2d_view tol_time(
          "tol time", problem_type::getNumberOfTimeODEs(kmcd, kmcdSurf, cstr), 2);
        real_type_1d_view tol_newton("tol newton", 2);

        real_type_2d_view fac(
          "fac", nBatch, problem_type::getNumberOfEquations(kmcd, kmcdSurf));
        //
        real_type_1d_view outlet_mass_flow("outlet mass flow", nBatch);
        auto outlet_mass_flow_host = Kokkos::create_mirror_view(outlet_mass_flow);

        /// tune tolerence
        {
          auto tol_time_host = Kokkos::create_mirror_view(tol_time);
          auto tol_newton_host = Kokkos::create_mirror_view(tol_newton);

          // const real_type atol_time = 1e-18;
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

        auto tadv_at_i_host = Kokkos::create_mirror_view(tadv_at_i);
        auto t_at_i_host = Kokkos::create_mirror_view(t_at_i);
        auto state_at_i_host = Kokkos::create_mirror_view(state_at_i);
        auto siteFraction_at_i_host =
          Kokkos::create_mirror_view(siteFraction_at_i);



        {
          real_type_2d_view EnthalpyMass("EnthalpyMass", 1, kmcd.nSpec);
          real_type_1d_view EnthalpyMixMass("EnthalpyMass Mixture", 1);

          const auto exec_space_instance = TChem::exec_space();
          TChem::EnthalpyMass::runDeviceBatch(exec_space_instance,
                                              team_size,
                                              vector_size,
                                              1,
                                              state,
                                              EnthalpyMass,
                                              EnthalpyMixMass,
                                              kmcd);
          cstr.EnthalpyIn = EnthalpyMixMass(0);
        }

        printf("Done Setting up CSTR reactor\n" );

        ordinal_type iter = 0;
        /// print of store QOI for the first sample
#if defined(TCHEM_EXAMPLE_SimpleSurface_QOI_PRINT)
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


          fprintf(fout, "\n");
          // save initial condition
          Kokkos::deep_copy(siteFraction_host, siteFraction);

          if (saveInitialCondition)
          {
            writeState(-1,
                     t_host,
                     dt_host,
                     state_host,
                     siteFraction_host,
                     fout);
          }




        /// team policy
        policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());

        const ordinal_type level = 1;
        const ordinal_type per_team_extent =
          TChem::TransientContStirredTankReactor::getWorkSpaceSize(kmcd, kmcdSurf);
        const ordinal_type per_team_scratch =
          TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
        policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

        real_type tsum(0);
        for (; iter < max_num_time_iterations && tsum <= tend*0.9999; ++iter) {

          if (isothermal)
          {
            TChem::IsothermalTransientContStirredTankReactor::runDeviceBatch(policy,
                                                   tol_newton,
                                                   tol_time,
                                                   fac,
                                                   tadv,
                                                   // input
                                                   state,
                                                   siteFraction,
                                                   // output
                                                   t,
                                                   dt,
                                                   state,
                                                   siteFraction,
                                                   outlet_mass_flow,
                                                   // kinetic info
                                                   kmcd,
                                                   kmcdSurf,
                                                   cstr);
          } else
          {
            TChem::TransientContStirredTankReactor::runDeviceBatch(policy,
                                                   tol_newton,
                                                   tol_time,
                                                   fac,
                                                   tadv,
                                                   // input
                                                   state,
                                                   siteFraction,
                                                   // output
                                                   t,
                                                   dt,
                                                   state,
                                                   siteFraction,
                                                   // kinetic info
                                                   kmcd,
                                                   kmcdSurf,
                                                   cstr);

          }


          //


          /// print of store QOI for the first sample
#if defined(TCHEM_EXAMPLE_SimpleSurface_QOI_PRINT)
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

          //
           {
            printf("save at iteration %d\n", iter);
            // time, sample, state
            // save time and dt

            Kokkos::deep_copy(dt_host, dt);
            Kokkos::deep_copy(t_host, t);
            Kokkos::deep_copy(state_host, state);
            Kokkos::deep_copy(siteFraction_host, siteFraction);

            writeState(iter,
                       t_host,
                       dt_host,
                       state_host,
                       siteFraction_host,
                       fout);
            //
            if (isothermal)
            {
              Kokkos::deep_copy(outlet_mass_flow_host, outlet_mass_flow);
              writeOutletMassFlowRate(iter,t_host,dt_host,outlet_mass_flow_host,fout_mass_flow);
            }
          }

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
    fprintf(fout_mass_flow, "\"dummy\":{} \n }\n");
    fclose(fout_mass_flow);
  }
  Kokkos::finalize();

  return 0;
}
