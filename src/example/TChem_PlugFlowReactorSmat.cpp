/* =====================================================================================
TChem version 2.1.0
Copyright (2020) NTESS
https://github.com/sandialabs/TChem

Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains 
certain rights in this software.

This file is part of TChem. TChem is open-source software: you can redistribute it
and/or modify it under the terms of BSD 2-Clause License
(https://opensource.org/licenses/BSD-2-Clause). A copy of the license is also
provided under the main directory

Questions? Contact Cosmin Safta at <csafta@sandia.gov>, or
           Kyungjoo Kim at <kyukim@sandia.gov>, or
           Oscar Diaz-Ibarra at <odiazib@sandia.gov>

Sandia National Laboratories, Livermore, CA, USA
===================================================================================== */


#include "TChem_PlugFlowReactorSmat.hpp"
#include "TChem_CommandLineParser.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_RateOfProgress.hpp"
#include "TChem_RateOfProgressSurface.hpp"
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
  std::string inputFileSurf(prefixPath + "inputSurfGas.dat");
  std::string inputFilevelocity(prefixPath + "inputVelocity.dat");
  std::string outputFile(prefixPath + "Smat_pfr.dat");

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
    "outputfile", "Output Smat file name e.g., Smat.dat", &outputFile);
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

    // input :: surface fraction vector, zk
    real_type_2d_view siteFraction("SiteFraction", nBatch, kmcdSurf.nSpec);
    // input :: velocity variables : velocity for  a PRF
    real_type_1d_view velocity("Velocity", nBatch);

    // density, temperature, gas.nSpe, velocity, surface.nSpe
    const auto Nvars = kmcd.nSpec + 3;
    const auto Nrg = kmcd.nReac;     // number of gas reactions
    const auto Nrs = kmcdSurf.nReac; // numger of surface reactions

    real_type_3d_view Smat("Smat_PlugflowreactorSmat", nBatch, Nvars, Nrg);
    real_type_3d_view Ssmat("Smat_PlugflowreactorSmat", nBatch, Nvars, Nrs);

    real_type_2d_view RoPFor("Gas_Forward_RateOfProgess", nBatch, Nrg);
    real_type_2d_view RoPRev("Gas_Reverse_RateOfProgess", nBatch, Nrg);

    real_type_2d_view RoPForSurf("Surface_Forward_RateOfProgess", nBatch, Nrs);
    real_type_2d_view RoPRevSurf("Surface_Reverse_RateOfProgess", nBatch, Nrs);

    /// create a mirror view to store input from a file
    auto state_host = Kokkos::create_mirror_view(state);

    /// create a mirror view to store input from a file
    auto siteFraction_host = Kokkos::create_mirror_view(siteFraction);

    /// create a mirror view to store input from a file
    auto velocity_host = Kokkos::create_mirror_view(velocity);

    real_type Area(0.00053);
    real_type Pcat(0.025977239243415308);

    pfr_data_type pfrd;
    pfrd.Area = Area; // m2
    pfrd.Pcat = Pcat; //

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
      auto velocity_host_at_0 = Kokkos::subview(velocity_host, 0);
      TChem::Test::read1DVector(inputFilevelocity, 1, velocity_host);
      TChem::Test::cloneView(velocity_host);
    }

    Kokkos::Impl::Timer timer;

    timer.reset();
    Kokkos::deep_copy(state, state_host);
    Kokkos::deep_copy(siteFraction, siteFraction_host);
    Kokkos::deep_copy(velocity, velocity_host);
    const real_type t_deepcopy = timer.seconds();

    timer.reset();

    TChem::PlugFlowReactorSmat::runDeviceBatch(nBatch,
                                               // inputs
                                               state,        // gas
                                               siteFraction, // surface
                                               velocity,     // PRF
                                               /// output
                                               Smat,
                                               Ssmat,
                                               kmcd,
                                               kmcdSurf,
                                               pfrd);

    TChem::RateOfProgress::runDeviceBatch(nBatch,
                                          state, // gas
                                          // output,
                                          RoPFor,
                                          RoPRev,
                                          kmcd);

    TChem::RateOfProgressSurface::runDeviceBatch(nBatch,
                                                 state,        // gas
                                                 siteFraction, // surface
                                                 // output,
                                                 RoPForSurf,
                                                 RoPRevSurf,
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
      auto Smat_host = Kokkos::create_mirror_view(Smat);
      Kokkos::deep_copy(Smat_host, Smat);

      /// all values are same (print only the first one)
      {
        auto Smat_host_at_0 =
          Kokkos::subview(Smat_host, 0, Kokkos::ALL(), Kokkos::ALL());
        TChem::Test::write2DMatrix(outputFile, Nvars, Nrg, Smat_host_at_0);
      }

      auto Ssmat_host = Kokkos::create_mirror_view(Ssmat);
      Kokkos::deep_copy(Ssmat_host, Ssmat);

      /// all values are same (print only the first one)
      {
        std::string outputFile(prefixPath + "Ssmat_pfr.dat");
        auto Ssmat_host_at_0 =
          Kokkos::subview(Ssmat_host, 0, Kokkos::ALL(), Kokkos::ALL());
        TChem::Test::write2DMatrix(outputFile, Nvars, Nrs, Ssmat_host_at_0);
      }

      auto RoPFor_host = Kokkos::create_mirror_view(RoPFor);
      Kokkos::deep_copy(RoPFor_host, RoPFor);

      /// all values are same (print only the first one)
      {
        std::string outputFile(prefixPath + "RoPForGas.dat");
        auto RoPFor_host_at_0 = Kokkos::subview(RoPFor_host, 0, Kokkos::ALL());
        TChem::Test::writeReactionRates(outputFile, Nrg, RoPFor_host_at_0);
      }

      auto RoPRev_host = Kokkos::create_mirror_view(RoPRev);
      Kokkos::deep_copy(RoPRev_host, RoPRev);

      /// all values are same (print only the first one)
      {
        std::string outputFile(prefixPath + "RoPRevGas.dat");
        auto RoPRev_host_at_0 = Kokkos::subview(RoPRev_host, 0, Kokkos::ALL());
        TChem::Test::writeReactionRates(outputFile, Nrg, RoPRev_host_at_0);
      }

      auto RoPForSurf_host = Kokkos::create_mirror_view(RoPForSurf);
      Kokkos::deep_copy(RoPForSurf_host, RoPForSurf);

      /// all values are same (print only the first one)
      {
        std::string outputFile(prefixPath + "RoPForSurf.dat");
        auto RoPForSurf_host_at_0 =
          Kokkos::subview(RoPForSurf_host, 0, Kokkos::ALL());
        TChem::Test::writeReactionRates(outputFile, Nrs, RoPForSurf_host_at_0);
      }

      auto RoPRevSurf_host = Kokkos::create_mirror_view(RoPRevSurf);
      Kokkos::deep_copy(RoPRevSurf_host, RoPRevSurf);

      /// all values are same (print only the first one)
      {
        std::string outputFile(prefixPath + "RoPRevSurf.dat");
        auto RoPRevSurf_host_at_0 =
          Kokkos::subview(RoPRevSurf_host, 0, Kokkos::ALL());
        TChem::Test::writeReactionRates(outputFile, Nrs, RoPRevSurf_host_at_0);
      }
    }

    printf("Done ...\n");
  }
  Kokkos::finalize();

  return 0;
}
