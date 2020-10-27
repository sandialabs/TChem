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
#include "TChem_GkSurfGas.hpp"
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
  std::string prefixPath("data/plug-flow-reactor/X/");
  std::string chemFile(prefixPath + "chem.inp");
  std::string thermFile(prefixPath + "therm.dat");
  std::string chemSurfFile(prefixPath + "chemSurf.inp");
  std::string thermSurfFile(prefixPath + "thermSurf.dat");

  std::string inputFile(prefixPath + "inputGas.dat");
  // std::string inputFileSurf(prefixPath + "inputSurfGas.dat");
  std::string outputFileGk(prefixPath + "gk.dat");
  std::string outputFileGkSurf(prefixPath + "gkSurf.dat");
  std::string outputFilehks(prefixPath + "hks.dat");
  std::string outputFilehksSurf(prefixPath + "hksSurf.dat");

  int nBatch(1);
  bool verbose(true);

  /// parse command line arguments
  TChem::CommandLineParser opts(
    "This example computes reaction rates with a given state vector");
  opts.set_option<std::string>(
    "chemfile", "Chem file name e.g., chem.inp", &chemFile);
  opts.set_option<std::string>(
    "thermfile", "Therm file name e.g., therm.dat", &thermFile);
  opts.set_option<std::string>(
    "chemSurffile", "Chem file name e.g., chem.inp", &chemSurfFile);
  opts.set_option<std::string>(
    "thermSurffile", "Therm file name e.g., therm.dat", &thermSurfFile);
  opts.set_option<std::string>(
    "inputfile", "Input state file name e.g., input.dat", &inputFile);
  opts.set_option<int>(
    "batchsize",
    "Batchsize the same state vector described in statefile is cloned",
    &nBatch);
  opts.set_option<bool>(
    "verbose", "If true, printout the first omega values", &verbose);

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse)
    return 0; // print help return

  Kokkos::initialize(argc, argv);
  {
    const bool detail = false;

    TChem::exec_space::print_configuration(std::cout, detail);
    TChem::host_exec_space::print_configuration(std::cout, detail);

    /// construct kmd and use the view for testing
    // TChem::KineticModelData kmd(chemFile, thermFile);
    // const auto kmcd = kmd.createConstData<TChem::exec_space>();

    // need to fix it
    TChem::KineticModelData kmdSurf(
      chemFile, thermFile, chemSurfFile, thermSurfFile);
    const auto kmcd =
      kmdSurf
        .createConstData<TChem::exec_space>(); // data struc with gas phase info
    const auto kmcdSurf =
      kmdSurf.createConstSurfData<TChem::exec_space>(); // data struc with
                                                        // surface phase info

    /// input: state vectors: temperature, pressure and mass fraction
    real_type_2d_view state(
      "StateVector", nBatch, TChem::Impl::getStateVectorSize(kmcd.nSpec));

    /// output: omega, reaction rates
    real_type_2d_view gk("gibbsGas", nBatch, kmcd.nSpec);
    real_type_2d_view gkSurf("gibbsSurf", nBatch, kmcdSurf.nSpec);

    real_type_2d_view hks("enthalpyGas", nBatch, kmcd.nSpec);
    real_type_2d_view hksSurf("enthalpySurf", nBatch, kmcdSurf.nSpec);

    /// create a mirror view to store input from a file
    auto state_host = Kokkos::create_mirror_view(state);

    /// input from a file; this is not necessary as the input is created
    /// by other applications.
    {
      // read gas
      auto state_host_at_0 = Kokkos::subview(state_host, 0, Kokkos::ALL());
      TChem::Test::readStateVector(inputFile, kmcd.nSpec, state_host_at_0);
      TChem::Test::cloneView(state_host);
    }

    Kokkos::Impl::Timer timer;

    timer.reset();
    Kokkos::deep_copy(state, state_host);
    const real_type t_deepcopy = timer.seconds();

    timer.reset();

    TChem::GkSurfGas::runDeviceBatch(
      nBatch, state, gk, gkSurf, hks, hksSurf, kmcd, kmcdSurf);

    Kokkos::fence(); /// timing purpose
    const real_type t_device_batch = timer.seconds();

    /// show time
    printf("Batch size %d, chemfile %s, thermfile %s, statefile %s\n",
           nBatch,
           chemFile.c_str(),
           thermFile.c_str(),
           inputFile.c_str());
    printf("---------------------------------------------------\n");
    printf("Time deep copy      %e [sec] %e [sec/sample]\n",
           t_deepcopy,
           t_deepcopy / real_type(nBatch));
    printf("Time reaction rates %e [sec] %e [sec/sample]\n",
           t_device_batch,
           t_device_batch / real_type(nBatch));

    //  create a mirror view of omeage (output) to export a file
    if (verbose) {

      auto gk_host = Kokkos::create_mirror_view(gk);
      Kokkos::deep_copy(gk_host, gk);

      auto gkSurf_host = Kokkos::create_mirror_view(gkSurf);
      Kokkos::deep_copy(gkSurf_host, gkSurf);

      auto hks_host = Kokkos::create_mirror_view(hks);
      Kokkos::deep_copy(hks_host, hks);

      auto hksSurf_host = Kokkos::create_mirror_view(hksSurf);
      Kokkos::deep_copy(hksSurf_host, hksSurf);

      /// all values are same (print only the first one)
      {

        auto gk_host_at_0 = Kokkos::subview(gk_host, 0, Kokkos::ALL());
        TChem::Test::writeReactionRates(outputFileGk, kmcd.nSpec, gk_host_at_0);

        auto gkSurf_host_at_0 = Kokkos::subview(gkSurf_host, 0, Kokkos::ALL());
        TChem::Test::writeReactionRates(
          outputFileGkSurf, kmcdSurf.nSpec, gkSurf_host_at_0);

        auto hks_host_at_0 = Kokkos::subview(hks_host, 0, Kokkos::ALL());
        TChem::Test::writeReactionRates(
          outputFilehks, kmcd.nSpec, hks_host_at_0);

        auto hksSurf_host_at_0 =
          Kokkos::subview(hksSurf_host, 0, Kokkos::ALL());
        TChem::Test::writeReactionRates(
          outputFilehksSurf, kmcdSurf.nSpec, hksSurf_host_at_0);
      }
    }

    printf("Done ... \n");
  }
  Kokkos::finalize();

  return 0;
}
