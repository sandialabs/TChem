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
#if defined(__TCHEM_DRIVER_GAS_ARRHENIUS_VARIATION__)
  std::string prefixPath("data/reaction-rates/");
#endif
  /// default inputs
#if defined(__TCHEM_DRIVER_GAS_NET_PRODUCTION_RATE__)
  std::string prefixPath("data/reaction-rates/");
#endif
#if defined(__TCHEM_DRIVER_GAS_TIME_INTEGRATION__)
  std::string prefixPath("data/ignition-zero-d/");
#endif
#if defined(__TCHEM_DRIVER_GAS_JACOBIAN_AND_RHS__)
  std::string prefixPath("data/ignition-zero-d/");
#endif
  std::string chemFile(prefixPath + "chem.inp");
  std::string thermFile(prefixPath + "therm.dat");
  std::string inputFile(prefixPath + "input.dat");
  std::string outputFile(prefixPath + "omega.dat");
  int nBatch(1), jacobian_interval(1);
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
  opts.set_option<int>("jacobian-interval", "Jacobians are evaluated in this interval during Newton solve", &jacobian_interval);
  opts.set_option<bool>(
			"verbose", "If true, printout the first omega values", &verbose);

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse)
    return 0; // print help return
  Kokkos::initialize(argc, argv);
#if defined(__TCHEM_DRIVER_GAS_NET_PRODUCTION_RATE__)
  {
    TChem::Driver tchem;
    tchem.createGasKineticModel(chemFile, thermFile);
    tchem.createGasKineticModelConstData();
    tchem.setNumberOfSamples(nBatch);
    tchem.createStateVector();
    //tchem.createNetProductionRatePerMass();
    tchem.showViews("After creating state vectors and net production rate per mass");

    const int nspec = tchem.getNumberOfSpecies();
    typename TChem::real_type_1d_view_host state("state", tchem.getLengthOfStateVector());
    TChem::Test::readStateVector(inputFile, nspec, state);
    for (int i=0;i<nBatch;++i)
      tchem.setStateVectorHost(i, state);
    tchem.showViews("After set state vector");

    tchem.computeGasNetProductionRatePerMassDevice();
    tchem.showViews("After compute net production rate per mass device");

    typename TChem::real_type_1d_const_view_host output;
    tchem.getGasNetProductionRatePerMassHost(0, output);
    tchem.showViews("After get net production rate per mass host");

    TChem::Test::writeReactionRates(outputFile, nspec, output);
  }
#endif

#if defined(__TCHEM_DRIVER_GAS_JACOBIAN_AND_RHS__)
  {
    TChem::Driver tchem;
    tchem.createGasKineticModel(chemFile, thermFile);
    tchem.createGasKineticModelConstData();
    tchem.setNumberOfSamples(nBatch);
    tchem.createStateVector();
    //tchem.createNetProductionRatePerMass();
    tchem.showViews("After creating state vectors and net production rate per mass");
    const int nspec = tchem.getNumberOfSpecies();
    typename TChem::real_type_1d_view_host state("state", tchem.getLengthOfStateVector());
    TChem::Test::readStateVector(inputFile, nspec, state);
    for (int i=0;i<nBatch;++i)
      tchem.setStateVectorHost(i, state);
    tchem.showViews("After set state vector");

    tchem.computeJacobianHomogeneousGasReactorDevice();
    tchem.computeRHS_HomogeneousGasReactorDevice();


    tchem.showViews("After compute net production rate per mass device");

    typename TChem::real_type_2d_const_view_host output;
    tchem.getJacobianHomogeneousGasReactorHost(0, output);
    tchem.showViews("After get net production rate per mass host");

    // TChem::Test::writeReactionRates(outputFile, nspec, output);

  }

#endif

#if defined(__TCHEM_DRIVER_GAS_TIME_INTEGRATION__)
  {
    TChem::Driver tchem;
    tchem.createGasKineticModel(chemFile, thermFile);
    tchem.createGasKineticModelConstData();

    tchem.setNumberOfSamples(nBatch);
    tchem.createStateVector();
    tchem.showViews("After creating state vectors and net production rate per mass");

    const int nspec = tchem.getNumberOfSpecies();
    typename TChem::real_type_1d_view_host state("state", tchem.getLengthOfStateVector());
    TChem::Test::readStateVector(inputFile, nspec, state);
    for (int i=0;i<nBatch;++i)
      tchem.setStateVectorHost(i, state);

    const real_type tbeg(0), tend(1), dtmin(1e-11), dtmax(1e-6);
    const int max_num_newton_iterations(20), num_time_iterations_per_interval(10);
    const real_type atol_newton(1e-8), rtol_newton(1e-5), atol_time(1e-12), rtol_time(1e-8);
    tchem.setTimeAdvanceHomogeneousGasReactor(tbeg, tend, dtmin, dtmax,
                                              jacobian_interval,
					      max_num_newton_iterations,
                                              num_time_iterations_per_interval,
					      atol_newton, rtol_newton,
					      atol_time, rtol_time);
    real_type tsum(0);
    TChem::real_type_1d_const_view_host t, dt, s;
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

#if defined(__TCHEM_DRIVER_GAS_ARRHENIUS_VARIATION__)
  {
    constexpr ordinal_type pre_exp_idx(0);
    constexpr ordinal_type temperature_coefficient_idx(1);
    constexpr ordinal_type activation_energy_idx(2);
    
    TChem::Driver tchem;
    tchem.createGasKineticModel(chemFile, thermFile);
    tchem.setNumberOfSamples(nBatch);

    // create an array of objects by cloning
    tchem.cloneGasKineticModel();
    
    //
    const int n_mod_reactions(2);
    // indx for reaction to be modified
    typename TChem::ordinal_type_1d_view_host reac_indices("modified reactions", n_mod_reactions);
    // changing reaction No 1 and 2 (indx starts from 0)
    reac_indices(0) = 1;reac_indices(1) = 2;

    typename TChem::real_type_3d_view_host factor("factor_pre_exp", nBatch, n_mod_reactions, 3);
    Kokkos::deep_copy(factor, real_type(1));
    
    for (int p = 0; p < nBatch; p++) {
      factor(p,0,pre_exp_idx) =1.5;
      factor(p,1,pre_exp_idx) =2.5;
    }

    tchem.modifyGasArrheniusForwardParameters(reac_indices, factor);
    tchem.createGasKineticModelConstData();
    
    tchem.createStateVector();
    tchem.showViews("After creating state vectors and net production rate per mass");

    const ordinal_type n_ver_reactions(3);
    typename TChem::ordinal_type_1d_view_host ver_reac_indices("verification modified reactions", n_ver_reactions);
    // check values for reactions No 1, 2, and 5 (indx starts from 0)
    ver_reac_indices(0) = 1;
    ver_reac_indices(1) = 2;
    ver_reac_indices(2) = 5;

    const ordinal_type imodel(0);
    typename TChem::real_type_1d_view_host pre_exp("pre_exp", n_ver_reactions);
    typename TChem::real_type_1d_view_host activation_energy("activation energy", n_ver_reactions);
    typename TChem::real_type_1d_view_host temperature_coefficient("temperature coefficients", n_ver_reactions);

    tchem.getGasArrheniusForwardParameter(imodel, ver_reac_indices,
					  pre_exp_idx, pre_exp);
    tchem.getGasArrheniusForwardParameter(imodel, ver_reac_indices,
					  temperature_coefficient_idx, temperature_coefficient);
    tchem.getGasArrheniusForwardParameter(imodel, ver_reac_indices,
					  activation_energy_idx, activation_energy);

    // orginal model
    typename TChem::real_type_1d_view_host pre_exp_org("pre_exp org", n_ver_reactions);
    typename TChem::real_type_1d_view_host activation_energy_org("activation energy org", n_ver_reactions);
    typename TChem::real_type_1d_view_host temperature_coefficient_org("temperature coefficients org", n_ver_reactions);

    tchem.getGasArrheniusForwardParameter(ver_reac_indices,
					  pre_exp_idx, pre_exp_org);
    tchem.getGasArrheniusForwardParameter(ver_reac_indices,
					  temperature_coefficient_idx, temperature_coefficient_org);
    tchem.getGasArrheniusForwardParameter(ver_reac_indices,
					  activation_energy_idx, activation_energy_org);

    printf("Checking parameters for Model %d ... \n", imodel );
    for (ordinal_type ireac = 0; ireac < ver_reac_indices.extent(0); ireac++) {
      printf("Reaction No %d\n", ver_reac_indices(ireac) );
      printf("Pre exponential: orginal %e modified %e factor %e \n", pre_exp_org(ireac),
	     pre_exp(ireac), pre_exp(ireac) / pre_exp_org(ireac) );

      printf("Temperature coefficients : orginal %e modified %e factor %e \n", temperature_coefficient_org(ireac),
	     temperature_coefficient(ireac), temperature_coefficient(ireac) / temperature_coefficient_org(ireac) );

      printf("Activation energy: orginal %e modified %e factor %e \n", activation_energy_org(ireac),
	     activation_energy(ireac), activation_energy(ireac) / activation_energy_org(ireac) );

    }
    printf("Done checking ... \n ");

    const int nspec = tchem.getNumberOfSpecies();
    typename TChem::real_type_1d_view_host state("state", tchem.getLengthOfStateVector());
    TChem::Test::readStateVector(inputFile, nspec, state);
    for (int i=0;i<nBatch;++i)
      tchem.setStateVectorHost(i, state);

    const real_type tbeg(0), tend(1), dtmin(1e-11), dtmax(1e-6);
    const int max_num_newton_iterations(20), num_time_iterations_per_interval(10);
    const real_type atol_newton(1e-8), rtol_newton(1e-5), atol_time(1e-12), rtol_time(1e-8);
    tchem.setTimeAdvanceHomogeneousGasReactor(tbeg, tend, dtmin, dtmax,
					      jacobian_interval,
					      max_num_newton_iterations,
					      num_time_iterations_per_interval,
					      atol_newton, rtol_newton,
					      atol_time, rtol_time);
    real_type tsum(0);
    TChem::real_type_1d_const_view_host t, dt, s;
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
