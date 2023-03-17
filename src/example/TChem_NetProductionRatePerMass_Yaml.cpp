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
// #include "TChem_NetProductionRatePerMass.hpp"
#include "TChem.hpp"
#include "TChem_CommandLineParser.hpp"
#include "TChem_Util.hpp"

using ordinal_type = TChem::ordinal_type;
using real_type = TChem::real_type;
using real_type_1d_view = TChem::real_type_1d_view;
using real_type_2d_view = TChem::real_type_2d_view;

int main(int argc, char *argv[]) {
#if defined(TCHEM_ENABLE_TPL_YAML_CPP)
  /// default inputs
  std::string prefixPath("data/reaction-rates/");
  std::string chemFile(prefixPath + "chem.inp");
  std::string inputFile(prefixPath + "input.dat");
  std::string outputFile(prefixPath + "omega.dat");
  std::string thermFile(prefixPath + "therm.dat");
  int nBatch(1), team_size(-1), vector_size(-1);
  ;
  bool verbose(true);
  bool useYaml(true);
  bool use_sample_format(false);

  /// parse command line arguments
  TChem::CommandLineParser opts("This example computes reaction rates with a given state vector");
  opts.set_option<std::string>("chemfile", "Chem file name e.g., chem.yaml", &chemFile);
  opts.set_option<std::string>("inputfile", "Input state file name e.g., input.dat", &inputFile);
  opts.set_option<std::string>("outputfile", "Output omega file name e.g., omega.dat", &outputFile);
  opts.set_option<std::string>("thermfile", "Therm file name e.g., therm.dat", &thermFile);
  opts.set_option<int>("batchsize", "Batchsize the same state vector described in statefile is cloned", &nBatch);
  //
  opts.set_option<int>("team_thread_size", "time thread size ", &team_size);
  //
  opts.set_option<int>("vector_thread_size", "vector thread size ", &vector_size);
  opts.set_option<bool>("verbose", "If true, printout the first omega values", &verbose);
  opts.set_option<bool>("use-yaml-parser", "If true, use yaml to parse input file", &useYaml);
  opts.set_option<bool>("use_sample_format", "If true, input file does not header or format", &use_sample_format);

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse)
    return 0; // print help return

  Kokkos::initialize(argc, argv);
  {
    const bool detail = false;

    TChem::exec_space().print_configuration(std::cout, detail);
    TChem::host_exec_space().print_configuration(std::cout, detail);
    const auto exec_space_instance = TChem::exec_space();

    /// construct kmd and use the view for testing
    TChem::KineticModelData kmd;

    if (useYaml) {
      kmd = TChem::KineticModelData(chemFile);
    } else {
      kmd = TChem::KineticModelData(chemFile, thermFile);
    }

    using device_type = typename Tines::UseThisDevice<exec_space>::type;

    const auto kmcd = TChem::createGasKineticModelConstData<device_type>(kmd);

    const ordinal_type stateVecDim = TChem::Impl::getStateVectorSize(kmcd.nSpec);

    real_type_2d_view_host state_host;

    if (use_sample_format) {
      // read gas
      const auto speciesNamesHost = Kokkos::create_mirror_view(kmcd.speciesNames);
      Kokkos::deep_copy(speciesNamesHost, kmcd.speciesNames);

      // get species molecular weigths
      const auto SpeciesMolecularWeights = Kokkos::create_mirror_view(kmcd.sMass);
      Kokkos::deep_copy(SpeciesMolecularWeights, kmcd.sMass);

      TChem::Test::readSample(inputFile, speciesNamesHost, SpeciesMolecularWeights, kmcd.nSpec, stateVecDim, state_host,
                              nBatch);

    } else {
      state_host = real_type_2d_view_host("state vector host", nBatch, stateVecDim);
      auto state_host_at_0 = Kokkos::subview(state_host, 0, Kokkos::ALL());
      TChem::Test::readStateVector(inputFile, kmcd.nSpec, state_host_at_0);
      TChem::Test::cloneView(state_host);
    }

    real_type_2d_view state("StateVector", nBatch, TChem::Impl::getStateVectorSize(kmcd.nSpec));
    /// output: omega, reaction rates
    real_type_2d_view omega("NetProductionRatePerMass", nBatch, kmcd.nSpec);

    using policy_type = typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;

    policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());
    const ordinal_type level = 1;

    if (team_size > 0 && vector_size > 0) {
      policy = policy_type(exec_space_instance, nBatch, team_size, vector_size);
    }

    const ordinal_type per_team_extent = NetProductionRatePerMass::getWorkSpaceSize(kmcd);
    ordinal_type per_team_scratch = TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
    policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

    Kokkos::Timer timer;

    timer.reset();
    Kokkos::deep_copy(state, state_host);
    const real_type t_deepcopy = timer.seconds();

    timer.reset();
    TChem::NetProductionRatePerMass::runDeviceBatch(policy, state, omega, kmcd);
    Kokkos::fence(); /// timing purpose
    const real_type t_device_batch = timer.seconds();

    {
      policy_type policy_kfwd_krev(exec_space_instance, nBatch, Kokkos::AUTO());
      const ordinal_type level = 1;

      const ordinal_type per_team_extent = TChem::KForwardReverse::getWorkSpaceSize(kmcd);
      ordinal_type per_team_scratch = TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
      policy_kfwd_krev.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
      real_type_2d_view kfor("NetProductionRatePerMass", nBatch, kmcd.nReac);
      real_type_2d_view krev("NetProductionRatePerMass", nBatch, kmcd.nReac);
      TChem::KForwardReverse::runDeviceBatch(policy_kfwd_krev, state, kfor, krev, kmcd);

      if (verbose) {
        auto kfor_host = Kokkos::create_mirror_view(kfor);
        Kokkos::deep_copy(kfor_host, kfor);

        auto krev_host = Kokkos::create_mirror_view(krev);
        Kokkos::deep_copy(krev_host, krev);

        auto kfor_host_at_0 = Kokkos::subview(kfor_host, 0, Kokkos::ALL());
        TChem::Test::writeReactionRates("kfwd_" + outputFile, kmcd.nReac, kfor_host_at_0);

        auto krev_host_at_0 = Kokkos::subview(krev_host, 0, Kokkos::ALL());
        TChem::Test::writeReactionRates("krev_" + outputFile, kmcd.nReac, krev_host_at_0);
      }
    }

    {

      policy_type policy_rop(exec_space_instance, nBatch, Kokkos::AUTO());
      const ordinal_type level = 1;

      const ordinal_type per_team_extent = TChem::RateOfProgress::getWorkSpaceSize(kmcd);
      ordinal_type per_team_scratch = TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
      policy_rop.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
      real_type_2d_view RoPFor("Rate of progress fwd", nBatch, kmcd.nReac);
      real_type_2d_view RoPRev("Rate of progress rev", nBatch, kmcd.nReac);

      TChem::RateOfProgress::runDeviceBatch(policy_rop, state, RoPFor, RoPRev, kmcd);

      if (verbose) {
        auto RoPFor_host = Kokkos::create_mirror_view(RoPFor);
        Kokkos::deep_copy(RoPFor_host, RoPFor);

        auto RoPRev_host = Kokkos::create_mirror_view(RoPRev);
        Kokkos::deep_copy(RoPRev_host, RoPRev);

        auto RoPFor_host_at_0 = Kokkos::subview(RoPFor_host, 0, Kokkos::ALL());
        TChem::Test::writeReactionRates("ropfwd_" + outputFile, kmcd.nReac, RoPFor_host_at_0);

        auto RoPRev_host_at_0 = Kokkos::subview(RoPRev_host, 0, Kokkos::ALL());
        TChem::Test::writeReactionRates("roprev_" + outputFile, kmcd.nReac, RoPRev_host_at_0);
      }
    }

    /// show time
    printf("---------------------------------------------------\n");
    printf("Testing Arguments: \n batch size %d\n chemfile %s\n inputfile %s\n outputfile %s\n verbose %s\n", nBatch,
           chemFile.c_str(), inputFile.c_str(), outputFile.c_str(), verbose ? "true" : "false");
    printf("---------------------------------------------------\n");
    printf("Time deep copy      %e [sec] %e [sec/sample]\n", t_deepcopy, t_deepcopy / real_type(nBatch));
    printf("Time reaction rates %e [sec] %e [sec/sample]\n", t_device_batch, t_device_batch / real_type(nBatch));

    /// create a mirror view of omeage (output) to export a file
    if (verbose) {
      auto omega_host = Kokkos::create_mirror_view(omega);
      Kokkos::deep_copy(omega_host, omega);

      /// all values are same (print only the first one)
      {
        auto omega_host_at_0 = Kokkos::subview(omega_host, 0, Kokkos::ALL());
        TChem::Test::writeReactionRates(outputFile, kmcd.nSpec, omega_host_at_0);
      }
    }
  }
  Kokkos::finalize();

#else
  printf("This example requires Yaml ...\n");
#endif

  return 0;
}
