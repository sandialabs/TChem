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

#if defined(TCHEM_TEST_DRIVER_NET_PRODUCTION_RATE)
  {
    TChem_createKineticModel(chemFile.c_str(), thermFile.c_str());
    TChem_setNumberOfSamples(nBatch);
    TChem_createStateVector();
    TChem_showAllViews("After creating state vectors and net production rate per mass");

    const int nspec = TChem_getNumberOfSpecies();
    const int lensv = TChem_getLengthOfStateVector();
    std::vector<real_type> state_std_vector(lensv);
    typename TChem::Driver::real_type_1d_view_host state(state_std_vector.data(), lensv);
    TChem::Test::readStateVector(inputFile, nspec, state);
    for (int i=0;i<nBatch;++i)
      TChem_setSingleStateVectorHost(i, state.data());

    TChem_showAllViews("After set state vector");
    
    TChem_computeNetProductionRatePerMassDevice();
    TChem_showAllViews("After compute net production rate per mass device");

    std::vector<real_type> output_std_vector(lensv);
    typename TChem::Driver::real_type_1d_view_host output(output_std_vector.data(), lensv);
    TChem_getSingleNetProductionRatePerMassHost(0, output.data());
    TChem_showAllViews("After get net production rate per mass host");
    
    TChem::Test::writeReactionRates(outputFile, nspec, output);
    TChem_freeKineticModel();
  }
#endif
  
#if defined(TCHEM_TEST_DRIVER_HOMOGENEOUS_BATCH_REACTOR)
  {
    TChem_createKineticModel(chemFile.c_str(), thermFile.c_str());    
    TChem_setNumberOfSamples(nBatch);
    TChem_createStateVector();
    TChem_showAllViews("After creating state vectors and net production rate per mass");

    const int nspec = TChem_getNumberOfSpecies();
    const int lensv = TChem_getLengthOfStateVector();
    std::vector<real_type> state_std_vector(lensv);
    typename TChem::Driver::real_type_1d_view_host state(state_std_vector.data(), lensv);
    TChem::Test::readStateVector(inputFile, nspec, state);
    for (int i=0;i<nBatch;++i)
      TChem_setSingleStateVectorHost(i, state.data());

    const real_type tbeg(0), tend(1), dtmin(1e-11), dtmax(1e-6);
    const int max_num_newton_iterations(20), num_time_iterations_per_interval(10);
    const real_type atol_newton(1e-8), rtol_newton(1e-5), atol_time(1e-12), rtol_time(1e-8);
    TChem_setTimeAdvanceHomogeneousGasReactor(tbeg, tend, dtmin, dtmax,
					      max_num_newton_iterations, num_time_iterations_per_interval,
					      atol_newton, rtol_newton,
					      atol_time, rtol_time);
    real_type tsum(0);
    std::vector<real_type> t(nBatch), dt(nBatch), s(lensv);
    for (int i=0;tsum<tend && i<1000;++i) {
      TChem_computeTimeAdvanceHomogeneousGasReactorDevice();
      TChem_getTimeStepHost(t.data());
      TChem_getTimeStepSizeHost(dt.data());
      TChem_getSingleStateVectorHost(0, s.data());

      printf("%e %e %e %e %e",
	     t[0],
	     dt[0],
	     s[0],
	     s[1],
	     s[2]);
      for (int k = 3, kend = s.size(); k < kend; ++k)
        printf(" %e", s[k]);
      printf("\n");
    }

  }
#endif

  return 0;
}
