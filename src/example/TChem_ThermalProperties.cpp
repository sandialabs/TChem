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
#include "TChem_CommandLineParser.hpp"
#include "TChem_EnthalpyMass.hpp"
#include "TChem_InternalEnergyMass.hpp"
#include "TChem_EntropyMass.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_SpecificHeatCapacityPerMass.hpp"
#include "TChem_SpecificHeatCapacityConsVolumePerMass.hpp"
#include "TChem_Util.hpp"

using ordinal_type = TChem::ordinal_type;
using real_type = TChem::real_type;
using real_type_1d_view = TChem::real_type_1d_view;
using real_type_2d_view = TChem::real_type_2d_view;

int
main(int argc, char* argv[])
{

  /// default inputs
  std::string prefixPath("data/");
  std::string chemFile(prefixPath + "chem.inp");
  std::string thermFile(prefixPath + "therm.dat");
  std::string inputFile(prefixPath + "sample.dat");
  // std::string outputFile(prefixPath + "CpMixMass.dat");
  bool verbose(true);

  int nBatch(1), team_size(-1), vector_size(-1);
  ;

  /// parse command line arguments
  TChem::CommandLineParser opts(
    "This example computes reaction rates with a given state vector");
  opts.set_option<std::string>(
    "chemfile", "Chem file name e.g., chem.inp", &chemFile);
  opts.set_option<std::string>(
    "thermfile", "Therm file name e.g., therm.dat", &thermFile);
  opts.set_option<std::string>(
    "inputfile", "Input state file name e.g., input.dat", &inputFile);
  // opts.set_option<std::string>("outputfile", "Output omega file name e.g.,
  // omega.dat", &outputFile);
  opts.set_option<int>(
    "batchsize",
    "Batchsize the same state vector described in statefile is cloned",
    &nBatch);
  opts.set_option<bool>(
    "verbose", "If true, printout the first omega values", &verbose);
  opts.set_option<int>("vector-size", "User defined vector size", &vector_size);
  opts.set_option<bool>(
    "verbose", "If true, printout the first Jacobian values", &verbose);

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse)
    return 0; // print help return

  Kokkos::initialize(argc, argv);
  {
    const bool detail = false;

    const auto exec_space_instance = TChem::exec_space();

    TChem::exec_space::print_configuration(std::cout, detail);
    TChem::host_exec_space::print_configuration(std::cout, detail);

    /// construct kmd and use the view for testing
    TChem::KineticModelData kmd(chemFile, thermFile);
    const auto kmcd = kmd.createConstData<TChem::exec_space>();

    const ordinal_type stateVecDim =
      TChem::Impl::getStateVectorSize(kmcd.nSpec);

    real_type_2d_view_host state_host;
    const auto speciesNamesHost = Kokkos::create_mirror_view(kmcd.speciesNames);
    Kokkos::deep_copy(speciesNamesHost, kmcd.speciesNames);
    {
      // get species molecular weigths
      const auto SpeciesMolecularWeights =
        Kokkos::create_mirror_view(kmcd.sMass);
      Kokkos::deep_copy(SpeciesMolecularWeights, kmcd.sMass);

      TChem::Test::readSample(inputFile,
                              speciesNamesHost,
                              SpeciesMolecularWeights,
                              kmcd.nSpec,
                              stateVecDim,
                              state_host,
                              nBatch);
    }
    real_type_2d_view state("StateVector Devices", nBatch, stateVecDim);

    // output:: CpMass
    real_type_2d_view CpMass("CpMass", nBatch, kmcd.nSpec);
    real_type_1d_view CpMixMass("CpMass Mixture", nBatch);

    real_type_1d_view CvMixMass("CvMass Mixture", nBatch);

    real_type_2d_view EnthalpyMass("EnthalpyMass", nBatch, kmcd.nSpec);
    real_type_1d_view EnthalpyMixMass("EnthalpyMass Mixture", nBatch);

    real_type_2d_view EntropyMass("EntropyMass", nBatch, kmcd.nSpec);
    real_type_1d_view EntropyMixMass("EntropyMass Mixture", nBatch);

    real_type_2d_view InternalEnergyMass("InternalEnergyMass", nBatch, kmcd.nSpec);
    real_type_1d_view InternalEnergyMixMass("InternalEnergyMass Mixture", nBatch);

    // compute heat capacity in mass base
    Kokkos::Impl::Timer timer;

    timer.reset();
    Kokkos::deep_copy(state, state_host);
    const real_type t_deepcopy = timer.seconds();

    using policy_type =
      typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;

    const ordinal_type level = 1;

    timer.reset();

    {

      policy_type policy_cp(exec_space_instance, nBatch, Kokkos::AUTO());

      const ordinal_type per_team_extent_cp =
        TChem::SpecificHeatCapacityPerMass::getWorkSpaceSize();
      ordinal_type per_team_scratch_cp =
        TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent_cp);
      policy_cp.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch_cp));

      TChem::SpecificHeatCapacityPerMass::runDeviceBatch(policy_cp,
                                                         state,
                                                         CpMass,
                                                         CpMixMass,
                                                         kmcd);
    }


    {

      const ordinal_type per_team_extent_h = EnthalpyMass::getWorkSpaceSize(kmcd);
      const ordinal_type per_team_scratch_h =
        Scratch<real_type_1d_view>::shmem_size(per_team_extent_h);

      policy_type policy_enthalpy(exec_space_instance, nBatch, Kokkos::AUTO());
      if (team_size > 0 && vector_size > 0) {
        policy_enthalpy = policy_type(exec_space_instance, nBatch, team_size, vector_size);
      }
      policy_enthalpy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch_h));

      TChem::EnthalpyMass::runDeviceBatch(policy_enthalpy,
                                          state,
                                          EnthalpyMass,
                                          EnthalpyMixMass,
                                          kmcd);
    }

    //

    {
      const ordinal_type per_team_extent_entropy = EntropyMass::getWorkSpaceSize(kmcd);
      const ordinal_type per_team_scratch_entropy =
        Scratch<real_type_1d_view>::shmem_size(per_team_extent_entropy);

      policy_type policy_entropy(exec_space_instance, nBatch, Kokkos::AUTO());
      if (team_size > 0 && vector_size > 0) {
        policy_entropy = policy_type(exec_space_instance, nBatch, team_size, vector_size);
      }

      policy_entropy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch_entropy));

      TChem::EntropyMass::runDeviceBatch(policy_entropy,
                                          state,
                                          EntropyMass,
                                          EntropyMixMass,
                                          kmcd);
    }

    //

    {

      const ordinal_type per_team_extent_int_energy = InternalEnergyMass::getWorkSpaceSize(kmcd);
      const ordinal_type per_team_scratch_int_energy =
        Scratch<real_type_1d_view>::shmem_size(per_team_extent_int_energy);

      policy_type policy_int_energy(exec_space_instance, nBatch, Kokkos::AUTO());
      if (team_size > 0 && vector_size > 0) {
        policy_int_energy = policy_type(exec_space_instance, nBatch, team_size, vector_size);
      }
      policy_int_energy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch_int_energy));

      TChem::InternalEnergyMass::runDeviceBatch(policy_int_energy,
                                          state,
                                          InternalEnergyMass,
                                          InternalEnergyMixMass,
                                          kmcd);

    }

    {

      const ordinal_type per_team_extent_cv =
        TChem::SpecificHeatCapacityConsVolumePerMass::getWorkSpaceSize(kmcd);
      const ordinal_type per_team_scratch_cv =
        Scratch<real_type_1d_view>::shmem_size(per_team_extent_cv);

      policy_type policy_cv(exec_space_instance, nBatch, Kokkos::AUTO());
      if (team_size > 0 && vector_size > 0) {
        policy_cv = policy_type(exec_space_instance, nBatch, team_size, vector_size);
      }
      policy_cv.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch_cv));


      TChem::SpecificHeatCapacityConsVolumePerMass::
      runDeviceBatch(policy_cv,
                     state,
                     CvMixMass,
                     kmcd);
    }




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

    /// create a mirror view of CpMixMass (output) to export a file
    if (verbose) {
      auto CpMixMass_host = Kokkos::create_mirror_view(CpMixMass);
      Kokkos::deep_copy(CpMixMass_host, CpMixMass);

      TChem::Test::write1DVector("CpMixMass.dat", CpMixMass_host);

      auto CvMixMass_host = Kokkos::create_mirror_view(CvMixMass);
      Kokkos::deep_copy(CvMixMass_host, CvMixMass);

      TChem::Test::write1DVector("CvMixMass.dat", CvMixMass_host);

      //
      auto EnthalpyMixMass_host = Kokkos::create_mirror_view(EnthalpyMixMass);
      Kokkos::deep_copy(EnthalpyMixMass_host, EnthalpyMixMass);

      TChem::Test::write1DVector("EnthalpyMixMass.dat", EnthalpyMixMass_host);


      auto EntropyMixMass_host = Kokkos::create_mirror_view(EntropyMixMass);
      Kokkos::deep_copy(EntropyMixMass_host, EntropyMixMass);

      TChem::Test::write1DVector("EntropyMixMass.dat", EntropyMixMass_host);

      auto InternalEnergyMixMass_host = Kokkos::create_mirror_view(InternalEnergyMixMass);
      Kokkos::deep_copy(InternalEnergyMixMass_host, InternalEnergyMixMass);

      TChem::Test::write1DVector("InternalEnergyMixMass.dat", InternalEnergyMixMass_host);
    }
  }
  Kokkos::finalize();

  return 0;
}
