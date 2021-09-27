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
#include "TChem_SourceTermToyProblem.hpp"
#include "TChem_CommandLineParser.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"

using ordinal_type = TChem::ordinal_type;
using real_type = TChem::real_type;
using real_type_1d_view = TChem::real_type_1d_view;
using real_type_2d_view = TChem::real_type_2d_view;

int
main(int argc, char* argv[])
{

  /// default inputs
  std::string prefixPath("data/toy-problem/");
  std::string chemFile(prefixPath + "chem.inp");
  std::string thermFile(prefixPath + "therm.dat");
  // std::string inputFile(prefixPath + "input.dat");
  std::string outputFile(prefixPath + "omega.dat");
  int nBatch(1);
  bool verbose(true);
  bool useYaml(false);

  /// parse command line arguments
  TChem::CommandLineParser opts(
    "This example computes reaction rates with a given state vector");
  opts.set_option<std::string>(
    "chemfile", "Chem file name e.g., chem.inp", &chemFile);
  opts.set_option<std::string>(
    "thermfile", "Therm file name e.g., therm.dat", &thermFile);
  // opts.set_option<std::string>(
  //   "inputfile", "Input state file name e.g., input.dat", &inputFile);
  opts.set_option<std::string>(
    "outputfile", "Output omega file name e.g., omega.dat", &outputFile);
  opts.set_option<int>(
    "batchsize",
    "Batchsize the same state vector described in statefile is cloned",
    &nBatch);
  opts.set_option<bool>(
    "verbose", "If true, printout the first omega values", &verbose);
  opts.set_option<bool>(
    "useYaml", "If true, use yaml to parse input file", &useYaml);

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse)
    return 0; // print help return

  Kokkos::initialize(argc, argv);
  {
    const bool detail = false;

    TChem::exec_space::print_configuration(std::cout, detail);
    TChem::host_exec_space::print_configuration(std::cout, detail);


    #if defined(TCHEM_ENABLE_TPL_YAML_CPP)
    TChem::KineticModelData kmd;
    if (useYaml) {
      kmd = TChem::KineticModelData(chemFile);
    } else {
      kmd = TChem::KineticModelData(chemFile, thermFile);
    }

    #else
    /// construct kmd and use the view for testing
    TChem::KineticModelData kmd(chemFile, thermFile);
    #endif

    using device_type      = typename Tines::UseThisDevice<exec_space>::type;
    const auto kmcd = TChem::createGasKineticModelConstData<device_type>(kmd);

    /// input: state vectors: temperature, pressure and mass fraction
    real_type_2d_view state(
      "StateVector", nBatch, kmcd.nSpec);

    /// output: omega, reaction rates
    real_type_2d_view omega("NetProductionRate", nBatch, kmcd.nSpec);

    real_type_1d_view theta("latitude", nBatch);
    real_type_1d_view lambda("longitude", nBatch);


    theta(0) = 20;
    lambda(0) = 300;

    /// create a mirror view to store input from a file
    auto state_host = Kokkos::create_mirror_view(state);

    /// input from a file; this is not necessary as the input is created
    /// by other applications.
    {
      auto state_host_at_0 = Kokkos::subview(state_host, 0, Kokkos::ALL());
      state_host_at_0(0) = 2.5e-7;
      state_host_at_0(1) = 1.8e-6;
      TChem::Test::cloneView(state_host);
    }

    Kokkos::Impl::Timer timer;

    timer.reset();
    Kokkos::deep_copy(state, state_host);
    const real_type t_deepcopy = timer.seconds();

    timer.reset();

    const auto exec_space_instance = TChem::exec_space();
    using policy_type =
      typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;

    /// team policy
    policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());

    const ordinal_type level = 1;
    const ordinal_type per_team_extent =
      TChem::SourceTermToyProblem::getWorkSpaceSize(kmcd);
    const ordinal_type per_team_scratch =
      TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
    policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

    TChem::SourceTermToyProblem::runDeviceBatch(policy, theta, lambda, state, omega, kmcd);



    Kokkos::fence(); /// timing purpose
    const real_type t_device_batch = timer.seconds();

    /// show time
    printf("---------------------------------------------------\n");
    printf("Testing Arguments: \n batch size %d\n chemfile %s\n thermfile %s\n outputfile %s\n verbose %s\n",
           nBatch,
           chemFile.c_str(),
           thermFile.c_str(),
           // inputFile.c_str(),
	   outputFile.c_str(),
	   verbose ? "true" : "false");
    printf("---------------------------------------------------\n");
    printf("Time deep copy      %e [sec] %e [sec/sample]\n",
           t_deepcopy,
           t_deepcopy / real_type(nBatch));
    printf("Time reaction rates %e [sec] %e [sec/sample]\n",
           t_device_batch,
           t_device_batch / real_type(nBatch));

    /// create a mirror view of omeage (output) to export a file
    if (verbose) {
      auto omega_host = Kokkos::create_mirror_view(omega);
      Kokkos::deep_copy(omega_host, omega);

      /// all values are same (print only the first one)
      {
        auto omega_host_at_0 = Kokkos::subview(omega_host, 0, Kokkos::ALL());
        TChem::Test::writeReactionRates(
          outputFile, kmcd.nSpec, omega_host_at_0);
      }
    }
  }
  Kokkos::finalize();

  return 0;
}
