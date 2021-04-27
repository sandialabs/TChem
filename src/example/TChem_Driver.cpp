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
#include "TChem.hpp"
#include "TChem_CommandLineParser.hpp"
#include "TChem_Driver.hpp"

using real_type = TChem::real_type;

int
main(int argc, char* argv[])
{
  //#define TCHEM_TEST_DRIVER_NET_PRODUCTION_RATE
#define TCHEM_TEST_DRIVER_HOMOGENEOUS_BATCH_REACTOR


  /// default inputs
#if defined(TCHEM_TEST_DRIVER_NET_PRODUCTION_RATE)
  std::string prefixPath("data/reaction-rates/");
#endif
#if defined(TCHEM_TEST_DRIVER_HOMOGENEOUS_BATCH_REACTOR)
  std::string prefixPath("data/ignition-zero-d/");
#endif
  std::string chemFile(prefixPath + "chem.inp");
  std::string thermFile(prefixPath + "therm.dat");
  std::string inputFile(prefixPath + "input.dat");
  std::string outputFile(prefixPath + "omega.dat");
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
    "outputfile", "Output omega file name e.g., omega.dat", &outputFile);
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
#if defined(TCHEM_TEST_DRIVER_NET_PRODUCTION_RATE)
  {
    TChem::Driver tchem;
    tchem.createKineticModel(chemFile, thermFile);
    tchem.setNumberOfSamples(nBatch);
    tchem.createStateVector();
    //tchem.createNetProductionRatePerMass();
    tchem.showViews("After creating state vectors and net production rate per mass");

    const int nspec = tchem.getNumberOfSpecies();
    typename TChem::Driver::real_type_1d_view_host state("state", tchem.getLengthOfStateVector());
    TChem::Test::readStateVector(inputFile, nspec, state);    
    for (int i=0;i<nBatch;++i)
      tchem.setStateVectorHost(i, state);
    tchem.showViews("After set state vector");
    
    tchem.computeNetProductionRatePerMassDevice();
    tchem.showViews("After compute net production rate per mass device");
    
    typename TChem::Driver::real_type_1d_const_view_host output;
    tchem.getNetProductionRatePerMassHost(0, output);
    tchem.showViews("After get net production rate per mass host");
    
    TChem::Test::writeReactionRates(outputFile, nspec, output);
  }
#endif
  
#if defined(TCHEM_TEST_DRIVER_HOMOGENEOUS_BATCH_REACTOR)
  {
    TChem::Driver tchem;
    tchem.createKineticModel(chemFile, thermFile);
    tchem.setNumberOfSamples(nBatch);
    tchem.createStateVector();
    tchem.showViews("After creating state vectors and net production rate per mass");

    const int nspec = tchem.getNumberOfSpecies();
    typename TChem::Driver::real_type_1d_view_host state("state", tchem.getLengthOfStateVector());
    TChem::Test::readStateVector(inputFile, nspec, state);    
    for (int i=0;i<nBatch;++i)
      tchem.setStateVectorHost(i, state);
    
    const real_type tbeg(0), tend(1), dtmin(1e-11), dtmax(1e-6);
    const int max_num_newton_iterations(20), num_time_iterations_per_interval(10);
    const real_type atol_newton(1e-8), rtol_newton(1e-5), atol_time(1e-12), rtol_time(1e-8);
    tchem.setTimeAdvanceHomogeneousGasReactor(tbeg, tend, dtmin, dtmax,
					      max_num_newton_iterations, num_time_iterations_per_interval,
					      atol_newton, rtol_newton,
					      atol_time, rtol_time);
    real_type tsum(0);
    TChem::Driver::real_type_1d_const_view_host t, dt, s;
    for (int i=0;tsum<tend && i<1000;++i) {
      tchem.computeTimeAdvanceHomogeneousGasReactorDevice();
      tchem.getTimeStepHost(t);
      tchem.getTimeStepSizeHost(dt);
      tchem.getStateVectorHost(0, s);

      printf("%e %e %e %e %e",
	     t(0),
	     dt(0),
	     s(0),
	     s(1),
	     s(2));
      for (int k = 3, kend = s.extent(0); k < kend; ++k)
        printf(" %e", s(k));
      printf("\n");
    }

  }
#endif
  Kokkos::finalize();
  return 0;
}
