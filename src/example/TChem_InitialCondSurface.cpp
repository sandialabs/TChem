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

#include "TChem_InitialCondSurface.hpp"

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

int
main(int argc, char* argv[])
{

  /// default inputs
  std::string prefixPath("data/plug-flow-reactor/X/");

  real_type atol_newton(1e-12), rtol_newton(1e-6);
  int max_num_newton_iterations(1000);

  int nBatch(1), team_size(-1), vector_size(-1);
  bool verbose(true);

  std::string chemFile("chem.inp");
  std::string thermFile("therm.dat");
  std::string chemSurfFile("chemSurf.inp");
  std::string thermSurfFile("thermSurf.dat");
  std::string inputFile("sample.dat");
  std::string inputFileSurf( "inputSurf.dat");
  bool use_prefixPath(true);

  /// parse command line arguments
  TChem::CommandLineParser opts(
    "This example computes reaction rates with a given state vector");
    opts.set_option<bool>(
        "use_prefixPath", "If true, input file are at the prefix path", &use_prefixPath);

    opts.set_option<std::string>(
      "prefixPath", "prefixPath e.g.,inputs/", &prefixPath);

    opts.set_option<std::string>
    ("chemfile", "Chem file name e.g., chem.inp",
    &chemFile);

    opts.set_option<std::string>
    ("thermfile", "Therm file name e.g., therm.dat", &thermFile);

    opts.set_option<std::string>
    ("chemSurffile","Chem file name e.g., chemSurf.inp",
     &chemSurfFile);

    opts.set_option<std::string>
    ("thermSurffile", "Therm file name e.g.,thermSurf.dat",
    &thermSurfFile);


    opts.set_option<std::string>
    ("samplefile", "Input state file name e.g., input.dat", &inputFile);
  opts.set_option<std::string>
    ("inputSurffile", "Input state file name e.g., inputSurfGas.dat", &inputFileSurf);

  opts.set_option<real_type>(
    "atol-newton", "Absolute tolerence used in newton solver", &atol_newton);
  opts.set_option<real_type>(
    "rtol-newton", "Relative tolerence used in newton solver", &rtol_newton);
  opts.set_option<int>("max-newton-iterations",
                       "Maximum number of newton iterations",
                       &max_num_newton_iterations);
  opts.set_option<int>(
    "batchsize",
    "Batchsize the same state vector described in statefile is cloned",
    &nBatch);
  opts.set_option<int>("team-size", "User defined team size", &team_size);
  opts.set_option<int>("vector-size", "User defined vector size", &vector_size);

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
      printf("Using a prefix path %s \n",prefixPath.c_str() );
    }

  Kokkos::initialize(argc, argv);
  {
    const bool detail = false;

    TChem::exec_space::print_configuration(std::cout, detail);
    TChem::host_exec_space::print_configuration(std::cout, detail);

    using device_type      = typename Tines::UseThisDevice<exec_space>::type;

    /// construct kmd and use the view for testing

    TChem::KineticModelData kmd(chemFile, thermFile, chemSurfFile, thermSurfFile);
    const auto kmcd = TChem::createGasKineticModelConstData<device_type>(kmd);
    const auto kmcdSurf = TChem::createSurfaceKineticModelConstData<device_type>(kmd);

    const ordinal_type stateVecDim =
      TChem::Impl::getStateVectorSize(kmcd.nSpec);

    //
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

      if (state_host.extent(0) != siteFraction_host.extent(0))
        std::logic_error("Error: number of sample is not valid");
    }

    real_type_2d_view siteFraction("SiteFraction", nBatch, kmcdSurf.nSpec);
    real_type_2d_view state("StateVector ", nBatch, stateVecDim);

    real_type_2d_view fac("fac", nBatch, kmcdSurf.nSpec);

    Kokkos::Timer timer;
    timer.reset();
    Kokkos::deep_copy(state, state_host);
    Kokkos::deep_copy(siteFraction, siteFraction_host);

    const real_type t_deepcopy = timer.seconds();

    {
      real_type_1d_view tol_newton("tol newton initial conditions", 2);
      auto tol_newton_host = Kokkos::create_mirror_view(tol_newton);

      tol_newton_host(0) = atol_newton;
      tol_newton_host(1) = rtol_newton;
      Kokkos::deep_copy(tol_newton, tol_newton_host);

      const auto exec_space_instance = TChem::exec_space();
      using policy_type =
        typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;

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
                                                tol_newton,
                                                max_num_newton_iterations,
                                                state,
                                                siteFraction, // input
                                                siteFraction, // output
                                                fac,
                                                kmcd,
                                                kmcdSurf);

      printf("Done with initial Condition for DAE system\n");
    }

    Kokkos::fence(); /// timing purpose
    const real_type t_device_batch = timer.seconds();
    printf("Time   %e [sec] %e [sec/sample]\n",
           t_device_batch,
           t_device_batch / real_type(nBatch));

    if (verbose) {
      Kokkos::deep_copy(siteFraction_host, siteFraction);
      auto siteFraction_host_at_0 =
        Kokkos::subview(siteFraction_host, 0, Kokkos::ALL());
      TChem::Test::write1DVector("InitialConditionSurface.dat",
                                 siteFraction_host_at_0);
    }
  }
  Kokkos::finalize();

  return 0;
}
