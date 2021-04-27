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
#include "TChem_Jacobian.hpp"
#include "TChem_CommandLineParser.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"

using ordinal_type = TChem::ordinal_type;
using real_type = TChem::real_type;
using real_type_1d_view = TChem::real_type_1d_view;
using real_type_2d_view = TChem::real_type_2d_view;
using real_type_3d_view = TChem::real_type_3d_view;

int
main(int argc, char* argv[])
{

  /// default inputs
  std::string prefixPath("data/jacobian/");
  std::string chemFile(prefixPath + "chem.inp");
  std::string thermFile(prefixPath + "therm.dat");
  std::string inputFile(prefixPath + "input.dat");
  std::string outputFile(prefixPath + "jacobian.dat");
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
    "inputfile", "Input state file name e.g., input.dat", &inputFile);
  opts.set_option<std::string>(
    "outputfile", "Output Jacobian file name e.g., jacobian.dat", &outputFile);
  opts.set_option<int>(
    "batchsize",
    "Batchsize the same state vector described in statefile is cloned",
    &nBatch);
  opts.set_option<bool>(
    "verbose", "If true, printout the first Jacobian values", &verbose);

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse)
    return 0; // print help return

  Kokkos::initialize(argc, argv);
  {
    const bool detail = false;

    TChem::exec_space::print_configuration(std::cout, detail);
    TChem::host_exec_space::print_configuration(std::cout, detail);

    /// construct kmd and use the view for testing
    TChem::KineticModelData kmd(chemFile, thermFile);
    const auto kmcd = kmd.createConstData<TChem::exec_space>();

    const ordinal_type jacDim = TChem::Impl::getStateVectorSize(kmcd.nSpec);

    /// input: state vectors: temperature, pressure and concentration
    real_type_2d_view state("StateVector", nBatch, jacDim);

    /// output: jacobian, nSpec+3 x nSpec+3 including temperature, pressure, rho
    real_type_3d_view jacobian("Jacobian", nBatch, jacDim, jacDim);

    /// create a mirror view to store input from a file
    auto state_host = Kokkos::create_mirror_view(state);

    /// input from a file; this is not necessary as the input is created
    /// by other applications.
    {
      auto state_host_at_i = Kokkos::subview(state_host, 0, Kokkos::ALL());
      TChem::Test::readStateVector(inputFile, kmcd.nSpec, state_host_at_i);
      TChem::Test::cloneView(state_host);
    }

    Kokkos::Impl::Timer timer;

    timer.reset();
    Kokkos::deep_copy(state, state_host);
    const real_type t_deepcopy = timer.seconds();

    timer.reset();
    TChem::Jacobian::runDeviceBatch(nBatch, state, jacobian, kmcd);

    Kokkos::fence(); /// timing purpose
    const real_type t_device_batch = timer.seconds();

    /// show time
    printf("Batch size %d, chemfile %s, thermfile %s, statefile %s\n",
           nBatch,
           chemFile.c_str(),
           thermFile.c_str(),
           inputFile.c_str());
    printf("---------------------------------------------------\n");
    printf("Time deep copy %e [sec] %e [sec/sample]\n",
           t_deepcopy,
           t_deepcopy / real_type(nBatch));
    printf("Time jacobian  %e [sec] %e [sec/sample]\n",
           t_device_batch,
           t_device_batch / real_type(nBatch));

    /// create a mirror view of jacobian (output) to export a file
    if (verbose) {
      auto jacobian_host = Kokkos::create_mirror_view(jacobian);
      Kokkos::deep_copy(jacobian_host, jacobian);

      /// all values are same (print only the first one)
      {
        auto jacobian_host_at_0 =
          Kokkos::subview(jacobian_host, 0, Kokkos::ALL(), Kokkos::ALL());
        TChem::Test::writeJacobian(outputFile, kmcd.nSpec, jacobian_host_at_0);
      }
    }
  }
  Kokkos::finalize();

  return 0;
}
