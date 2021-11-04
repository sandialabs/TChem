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
#include "TChem_SurfaceRHS.hpp"
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
  // std::string prefixPath("data/plug-flow-reactor/PT/");
  std::string prefixPath("data/plug-flow-reactor/X/");
  std::string chemFile(prefixPath + "chem.inp");
  std::string thermFile(prefixPath + "therm.dat");
  std::string chemSurfFile(prefixPath + "chemSurf.inp");
  std::string thermSurfFile(prefixPath + "thermSurf.dat");
  std::string inputFile(prefixPath + "inputGas.dat");
  std::string inputFileSurf(prefixPath + "inputSurfGas.dat");
  std::string inputFilevelocity(prefixPath + "inputVelocity.dat");
  std::string outputFile(prefixPath + "SurfaceRHS.dat");

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
    "inputfile", "Input state file name e.g., inputGas.dat", &inputFile);
  opts.set_option<std::string>("inputfile",
                               "Input state file name e.g., inputSurfGas.dat",
                               &inputFileSurf);
  opts.set_option<std::string>("inputfile",
                               "Input state file name e.g., inputVelocity.dat",
                               &inputFilevelocity);
  opts.set_option<std::string>(
    "outputfile", "Output rhs file name e.g., SurfaceRHS.dat", &outputFile);
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

    using device_type      = typename Tines::UseThisDevice<exec_space>::type;

    // need to fix it
    TChem::KineticModelData kmd(
      chemFile, thermFile, chemSurfFile, thermSurfFile);
    const auto kmcd =
      TChem::createGasKineticModelConstData<device_type>(kmd); // data struc with gas phase info
    const auto kmcdSurf =
      TChem::createSurfaceKineticModelConstData<device_type>(kmd); // data struc with
                                                        // surface phase info

    /// input: state vectors: temperature, pressure and mass fraction
    real_type_2d_view state(
      "StateVector", nBatch, TChem::Impl::getStateVectorSize(kmcd.nSpec));

    // input :: surface fraction vector, zk
    real_type_2d_view siteFraction("SiteFraction", nBatch, kmcdSurf.nSpec);
    // input :: velocity variables : velocity for  a PRF
    // real_type_1d_view velocity("Velocity", nBatch);
    /// output: rhs for surface reactions
    real_type_2d_view rhs("SurfaceRHS", nBatch, kmcdSurf.nSpec);
    // density, temperature, gas.nSpe, velocity, surface.nSpe

    /// create a mirror view to store input from a file
    auto state_host = Kokkos::create_mirror_view(state);

    /// create a mirror view to store input from a file
    auto siteFraction_host = Kokkos::create_mirror_view(siteFraction);

    /// create a mirror view to store input from a file
    // auto velocity_host = Kokkos::create_mirror_view(velocity);

    /// input from a file; this is not necessary as the input is created
    /// by other applications.
    {
      // read gas
      auto state_host_at_0 = Kokkos::subview(state_host, 0, Kokkos::ALL());
      TChem::Test::readStateVector(inputFile, kmcd.nSpec, state_host_at_0);
      TChem::Test::cloneView(state_host);
      // read surface
      auto siteFraction_host_at_0 =
        Kokkos::subview(siteFraction_host, 0, Kokkos::ALL());
      TChem::Test::readSiteFraction(
        inputFileSurf, kmcdSurf.nSpec, siteFraction_host_at_0);
      TChem::Test::cloneView(siteFraction_host);
      // read velocity (velocity)
      // auto velocity_host_at_0 = Kokkos::subview(velocity_host, 0);
      // TChem::Test::read1DVector(inputFilevelocity, 1, velocity_host);
      // TChem::Test::cloneView(velocity_host);
    }

    Kokkos::Timer timer;

    timer.reset();
    Kokkos::deep_copy(state, state_host);
    Kokkos::deep_copy(siteFraction, siteFraction_host);
    // Kokkos::deep_copy(velocity, velocity_host);
    const real_type t_deepcopy = timer.seconds();

    timer.reset();

    TChem::SurfaceRHS::runDeviceBatch(nBatch,
                                      // inputs
                                      state,        // gas
                                      siteFraction, // surface
                                      // ouputs
                                      rhs,
                                      // data
                                      kmcd,
                                      kmcdSurf);

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

    //  create a mirror view of rhs (output) to export a file
    if (verbose) {
      auto rhs_host = Kokkos::create_mirror_view(rhs);
      Kokkos::deep_copy(rhs_host, rhs);

      /// all values are same (print only the first one)
      {
        auto rhs_host_at_0 = Kokkos::subview(rhs_host, 0, Kokkos::ALL());
        TChem::Test::writeReactionRates(
          outputFile, kmcdSurf.nSpec, rhs_host_at_0);
      }
    }

    printf("Done ...\n");
  }
  Kokkos::finalize();

  return 0;
}
