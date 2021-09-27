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
#include "TChem_NetProductionRateSurfacePerMass.hpp"
#include "TChem_CommandLineParser.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_NetProductionRatePerMass.hpp"
#include "TChem_Util.hpp"

using ordinal_type = TChem::ordinal_type;
using real_type = TChem::real_type;
using real_type_1d_view = TChem::real_type_1d_view;
using real_type_2d_view = TChem::real_type_2d_view;

int
main(int argc, char* argv[])
{

  /// default inputs
    #if defined(TCHEM_ENABLE_TPL_YAML_CPP)


  std::string prefixPath("data/reaction-rates-surfaces/PT/");
  std::string chemFile(prefixPath + "chem.inp");
  std::string thermFile(prefixPath + "therm.dat");
  std::string chemSurfFile(prefixPath + "chemSurf.inp");
  std::string thermSurfFile(prefixPath + "thermSurf.dat");
  std::string inputFile(prefixPath + "inputGas.dat");
  std::string inputFileSurf(prefixPath + "inputSurfGas.dat");
  std::string outputFile(prefixPath + "omega.dat");
  std::string outputFileGasSurf(prefixPath + "omegaGasSurf.dat");
  std::string outputFileSurf(prefixPath + "omegaSurf.dat");

  int nBatch(1);
  bool verbose(true);
  bool useYaml(true);
  bool use_sample_format(false);

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
    "thermSurffile", "Therm file name e.g., thermSurf.dat", &thermSurfFile);
  opts.set_option<std::string>(
    "inputfile", "Input state file name e.g., input.dat", &inputFile);
  opts.set_option<std::string>(
    "inputfileSurf", "Input state file name e.g., input.dat", &inputFileSurf);
  opts.set_option<std::string>(
    "outputfile", "Output omega file name e.g., omega.dat", &outputFile);
  opts.set_option<std::string>(
    "outputfileGasSurf",
    "Output omega gas from surface file name e.g., omegaGasSurf.dat",
    &outputFileGasSurf);
  opts.set_option<std::string>("outputfileSurf",
                               "Output omega file name e.g., omegaGasSurf.dat",
                               &outputFileSurf);
  opts.set_option<int>(
    "batchsize",
    "Batchsize the same state vector described in statefile is cloned",
    &nBatch);
  opts.set_option<bool>(
    "verbose", "If true, printout the first omega values", &verbose);
  //
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


    /// construct kmd and use the view for testing
    TChem::KineticModelData kmd;

    if (useYaml) {
      kmd = TChem::KineticModelData(chemFile, true);
    } else {
      kmd = TChem::KineticModelData(chemFile, thermFile, chemSurfFile, thermSurfFile);
    }

    using device_type      = typename Tines::UseThisDevice<exec_space>::type;
    const auto kmcd = TChem::createGasKineticModelConstData<device_type>(kmd);
    const auto kmcdSurf =
      TChem::createSurfaceKineticModelConstData<device_type>(kmd); // data struc with

    /// input: state vectors: temperature, pressure and mass fraction
    real_type_2d_view state(
       "StateVector", nBatch, TChem::Impl::getStateVectorSize(kmcd.nSpec));

    // input :: surface fraction vector, zk
    real_type_2d_view siteFraction("Site fraction", nBatch, kmcdSurf.nSpec);
    /// output: omega, reaction rates
    real_type_2d_view omega("NetProductionRatePerMass", nBatch, kmcd.nSpec);

    /// output: omega, reaction rates from surface (gas species)
    real_type_2d_view omegaGasSurf("ReactionRatesGasSurf", nBatch, kmcd.nSpec);

    /// output: omega, reaction rates from surface (surface species)
    real_type_2d_view omegaSurf("ReactionRatesSurf", nBatch, kmcdSurf.nSpec);

    real_type_2d_view_host state_host;
    real_type_2d_view_host siteFraction_host;

    const ordinal_type stateVecDim =
      TChem::Impl::getStateVectorSize(kmcd.nSpec);


    if (use_sample_format){
      // read gas
      const auto speciesNamesHost = Kokkos::create_mirror_view(kmcd.speciesNames);
      Kokkos::deep_copy(speciesNamesHost, kmcd.speciesNames);

      // get species molecular weigths
      const auto SpeciesMolecularWeights =
        Kokkos::create_mirror_view(kmcd.sMass);
      Kokkos::deep_copy(SpeciesMolecularWeights, kmcd.sMass);

      // get names of species on host
      const auto SurfSpeciesNamesHost =
        Kokkos::create_mirror_view(kmcdSurf.speciesNames);
      Kokkos::deep_copy(SurfSpeciesNamesHost, kmcdSurf.speciesNames);

      TChem::Test::readSample(inputFile,
                              speciesNamesHost,
                              SpeciesMolecularWeights,
                              kmcd.nSpec,
                              stateVecDim,
                              state_host,
                              nBatch);
      //
      TChem::Test::readSurfaceSample(inputFileSurf,
                                     SurfSpeciesNamesHost,
                                     kmcdSurf.nSpec,
                                     siteFraction_host,
                                     nBatch);

      if (state_host.extent(0) != siteFraction_host.extent(0) )
        std::logic_error("Error: number of sample is not valid");

    } else{
      state_host = real_type_2d_view_host("state vector host", nBatch, stateVecDim);
      auto state_host_at_0 = Kokkos::subview(state_host, 0, Kokkos::ALL());
      TChem::Test::readStateVector(inputFile, kmcd.nSpec, state_host_at_0);
      TChem::Test::cloneView(state_host);

      siteFraction_host = real_type_2d_view_host("state vector host", nBatch, kmcdSurf.nSpec);
      auto siteFraction_host_at_0 = Kokkos::subview(siteFraction_host, 0, Kokkos::ALL());
      TChem::Test::readSiteFraction(inputFileSurf, kmcdSurf.nSpec, siteFraction_host_at_0);
      TChem::Test::cloneView(siteFraction_host);

    }

    Kokkos::Impl::Timer timer;

    timer.reset();
    Kokkos::deep_copy(state, state_host);
    Kokkos::deep_copy(siteFraction, siteFraction_host);
    const real_type t_deepcopy = timer.seconds();

    timer.reset();
    TChem::NetProductionRatePerMass::runDeviceBatch(state, omega, kmcd);
    TChem::NetProductionRateSurfacePerMass::runDeviceBatch(state, siteFraction, omegaGasSurf, omegaSurf, kmcd, kmcdSurf);
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
      auto omega_host = Kokkos::create_mirror_view(omega);
      Kokkos::deep_copy(omega_host, omega);

      auto omegaGasSurf_host = Kokkos::create_mirror_view(omegaGasSurf);
      Kokkos::deep_copy(omegaGasSurf_host, omegaGasSurf);

      auto omegaSurf_host = Kokkos::create_mirror_view(omegaSurf);
      Kokkos::deep_copy(omegaSurf_host, omegaSurf);

      /// all values are same (print only the first one)
      {
        auto omega_host_at_0 = Kokkos::subview(omega_host, 0, Kokkos::ALL());
        TChem::Test::writeReactionRates(
          outputFile, kmcd.nSpec, omega_host_at_0);

        auto omegaGasSurf_host_at_0 =
          Kokkos::subview(omegaGasSurf_host, 0, Kokkos::ALL());
        TChem::Test::writeReactionRates(
          outputFileGasSurf, kmcd.nSpec, omegaGasSurf_host_at_0);

        auto omegaSurf_host_at_0 =
          Kokkos::subview(omegaSurf_host, 0, Kokkos::ALL());
        TChem::Test::writeReactionRates(
          outputFileSurf, kmcdSurf.nSpec, omegaSurf_host_at_0);
      }
    }

    printf("Done \n");
  }
  Kokkos::finalize();

  #else
   printf("This example requires Yaml ...\n" );
  #endif

  return 0;
}
