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

using ordinal_type = TChem::ordinal_type;
using real_type = TChem::real_type;

using real_type_0d_view = TChem::real_type_0d_view;
using real_type_1d_view = TChem::real_type_1d_view;
using real_type_2d_view = TChem::real_type_2d_view;

using real_type_0d_view_host = TChem::real_type_0d_view_host;
using real_type_1d_view_host = TChem::real_type_1d_view_host;
using real_type_2d_view_host = TChem::real_type_2d_view_host;


// #define TCHEM_EXAMPLE_IGNITIONZEROD_QOI_PRINT

int
main(int argc, char* argv[])
{
  #if defined(TCHEM_ENABLE_TPL_YAML_CPP)
  /// default inputs
  std::string prefixPath("data");
  int nBatch(1), team_size(-1), vector_size(-1);
  bool verbose(true), use_sample_format(false);;
  bool isothermal(false);

  real_type mdotIn(3.596978981250784e-06);
  real_type Vol(0.00013470);
  real_type Acat (0.0013074);

  int number_of_algebraic_constraints(0);
  int poisoning_species_idx(-1);
  int ivar(0);

  std::string chemFile("chem.inp");
  std::string thermFile("therm.dat");
  std::string inputFile("sample.dat");
  std::string inputFileParamModifiers("ParameterModifiers.dat");
  std::string chemSurfFile("chemSurf.inp");
  std::string thermSurfFile("thermSurf.dat");
  std::string inputFileSurf( "inputSurf.dat");
  std::string outputFile("rhs_CSTR.dat");
  std::string inputFile_inlet("sample_ref.dat");

  /// parse command line arguments
  TChem::CommandLineParser opts(
    "This example computes the rhs of an isothermal Transient Cont Stirred Tank Reactor");
  opts.set_option<std::string>(
      "inputs-path", "prefixPath e.g.,inputs/", &prefixPath);
  opts.set_option<real_type>("catalytic-area", "Catalytic area [m2]", &Acat);
  opts.set_option<real_type>("reactor-volume", "Reactor Volumen [m3]", &Vol);
  opts.set_option<real_type>("inlet-mass-flow", "Inlet mass flow rate [kg/s]", &mdotIn);
  opts.set_option<bool>("isothermal", "if True, reaction is isotermic", &isothermal);

  opts.set_option<std::string>
  ("chemfile", "Chem file name e.g., chem.inp",&chemFile);
  opts.set_option<std::string>
  ("thermfile", "Therm file namee.g., therm.dat", &thermFile);
  opts.set_option<std::string>
  ("samplefile", "Input state file name e.g.,input.dat", &inputFile);
  opts.set_option<std::string>
  ("samplefile-inlet", "Input state file name e.g.,input.dat", &inputFile_inlet);
  opts.set_option<std::string>
  ("input-file-param-modifiers", "Input state file name e.g.,input.dat", &inputFileParamModifiers);
  opts.set_option<std::string>
  ("chemfile", "Chem file name of gas phase e.g., chem.inp", &chemFile);
  opts.set_option<std::string>
  ("thermfile", "Therm file name of gas phase  e.g., therm.dat", &thermFile);
  opts.set_option<std::string>
  ("surf-chemfile","Chem file name of surface phase e.g., chemSurf.inp",
   &chemSurfFile);
  opts.set_option<std::string>
  ("surf-thermfile", "Therm file name of surface phase e.g.,thermSurf.dat",
  &thermSurfFile);
  opts.set_option<std::string>
  ("samplefile", "Input state file name of gas phase e.g., input.dat", &inputFile);
  opts.set_option<std::string>
  ("surf-inputfile", "Input state file name of surface e.g., inputSurfGas.dat", &inputFileSurf);
  opts.set_option<std::string>
  ("input-file-param-modifiers", "Input state file name e.g.,input.dat", &inputFileParamModifiers);
  opts.set_option<bool>(
    "verbose", "If true, printout the first Jacobian values", &verbose);
  opts.set_option<int>("index-poisoning-species",
                       "catalysis deactivation, index for species",
                       &poisoning_species_idx);
  opts.set_option<int>("team-size", "User defined team size", &team_size);
  opts.set_option<int>("vector-size", "User defined vector size", &vector_size);
  opts.set_option<int>("var-index", "Save rhs of variable with index var-index", &ivar);
  opts.set_option<std::string>("outputfile",
  "Output file name e.g., CSTRSolution.dat", &outputFile);
  opts.set_option<bool>(
    "use_sample_format", "If true, input file has header or format", &use_sample_format);
  //
  opts.set_option<int>(
    "batchsize",
    "Batchsize the same state vector described in statefile is cloned",
    &nBatch);


  const bool r_parse = opts.parse(argc, argv);
  if (r_parse)
    return 0; // print help return


  Kokkos::initialize(argc, argv);
  {

    const bool detail = false;

    TChem::exec_space::print_configuration(std::cout, detail);
    TChem::host_exec_space::print_configuration(std::cout, detail);
    using device_type      = typename Tines::UseThisDevice<exec_space>::type;
    using host_device_type      = typename Tines::UseThisDevice<host_exec_space>::type;
    //
    TChem::KineticModelData kmd(
      chemFile, thermFile, chemSurfFile, thermSurfFile);
    const auto kmcd = TChem::createGasKineticModelConstData<device_type>(kmd);
    const auto kmcdSurf =
      TChem::createSurfaceKineticModelConstData<device_type>(kmd); // data struc with
    const auto kmcd_host = TChem::createGasKineticModelConstData<host_device_type>(kmd);
    const auto kmcdSurf_host =
      TChem::createSurfaceKineticModelConstData<host_device_type>(kmd);

    printf("Number of Species %d \n", kmcd.nSpec);
    printf("Number of Reactions %d \n", kmcd.nReac);

    const ordinal_type stateVecDim =
      TChem::Impl::getStateVectorSize(kmcd.nSpec);

    real_type_2d_view_host state_host;
    real_type_2d_view_host site_fraction_host;
    real_type_2d_view_host state_host_inlet;



    if (use_sample_format){
      auto speciesNamesHost = kmcd_host.speciesNames;
      auto SurfSpeciesNamesHost = kmcdSurf_host.speciesNames;
      // get species molecular weigths
      auto SpeciesMolecularWeights =kmcd_host.sMass;

      TChem::Test::readSample(inputFile,
                              speciesNamesHost,
                              SpeciesMolecularWeights,
                              kmcd.nSpec,
                              stateVecDim,
                              state_host,
                              nBatch);

      // read inlet condition
      ordinal_type nBatch_ref =0;
      TChem::Test::readSample(inputFile_inlet,
                              speciesNamesHost,
                              SpeciesMolecularWeights,
                              kmcd.nSpec,
                              stateVecDim,
                              state_host_inlet,
                              nBatch_ref);

      TChem::Test::readSurfaceSample(inputFileSurf,
                                     SurfSpeciesNamesHost,
                                     kmcdSurf.nSpec,
                                     site_fraction_host,
                                     nBatch);


      if (state_host.extent(0) != site_fraction_host.extent(0) )
        std::logic_error("Error: number of sample is not valid");
    } else {
      // read gas phase : temp, pressure, density, and mass fraction
      state_host = real_type_2d_view_host("state vector host", nBatch, stateVecDim);
      auto state_host_at_0 = Kokkos::subview(state_host, 0, Kokkos::ALL());
      TChem::Test::readStateVector(inputFile, kmcd.nSpec, state_host_at_0);
      TChem::Test::cloneView(state_host);

      //read inlet conditions CSTR
      state_host_inlet = real_type_2d_view_host("state vector host ref ", 1, stateVecDim);
      auto state_host_inlet_at_0 = Kokkos::subview(state_host_inlet, 0, Kokkos::ALL());
      TChem::Test::readStateVector(inputFile_inlet, kmcd.nSpec, state_host_inlet_at_0);

      // read surface interface: site fractions
      site_fraction_host = real_type_2d_view_host("site fraction vector host", nBatch, kmcdSurf.nSpec);
      auto site_fraction_host_at_0 = Kokkos::subview(site_fraction_host, 0, Kokkos::ALL());
      TChem::Test::readVector(inputFileSurf, kmcdSurf.nSpec, site_fraction_host_at_0);
      TChem::Test::cloneView(site_fraction_host);
    }
    printf("Number of Samples %d\n", nBatch );

    real_type_2d_view site_fraction("site_fraction", nBatch, kmcdSurf.nSpec);
    real_type_2d_view state("StateVector Devices", nBatch, stateVecDim);
    Kokkos::deep_copy(state, state_host);
    Kokkos::deep_copy(site_fraction, site_fraction_host);

    real_type_2d_view state_inlet("StateVector Devices ref", 1, stateVecDim);
    Kokkos::deep_copy(state_inlet, state_host_inlet);

    const YAML::Node inputFile = YAML::LoadFile(inputFileParamModifiers);


    /// different sample would have a different kinetic model
    //gas phase
    auto kmds = kmd.clone(nBatch);
    if (inputFile["Gas"])
    {
      printf("reading gas phase ...\n");
      constexpr ordinal_type gas(0);
      const auto DesignOfExperimentGas = inputFile["Gas"]["DesignOfExperiment"];
      TChem::modifyArrheniusForwardParameter(kmds, gas, DesignOfExperimentGas);
    }

    auto kmcds = TChem::createGasKineticModelConstData<device_type>(kmds);

    //surface phase
    if (inputFile["Surface"])
    {
      printf("reading surface phase ...\n");
      constexpr ordinal_type surface(1);
      const auto DesignOfExperimentSurface = inputFile["Surface"]["DesignOfExperiment"];
      TChem::modifyArrheniusForwardParameter(kmds, surface, DesignOfExperimentSurface);
    }

    auto kmcdSurfs = TChem::createSurfaceKineticModelConstData<device_type>(kmds);

    if (inputFile["Gas"])
    {
      TChem::Test::printParametersModelVariation("Gas", kmcd, kmcds, inputFile);
    }

    if (inputFile["Surface"])
    {
      TChem::Test::printParametersModelVariation("Surface", kmcdSurf, kmcdSurfs, inputFile);
    }

    const auto exec_space_instance = TChem::exec_space();

    using policy_type =
      typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;

    //
    printf("Setting up CSTR reactor\n");
    TransientContStirredTankReactorData<device_type> cstr;
    {
      cstr.mdotIn = mdotIn; // inlet mass flow kg/s
      cstr.Vol    = Vol; // volumen of reactor m3
      cstr.Acat   = Acat; // Catalytic area m2: chemical active area
      cstr.pressure = state_host_inlet(0,1);
      cstr.isothermal = 1;
      if (isothermal) cstr.isothermal = 0; // 0 constant temperature
      // cstr.temperature = state_host(0, 2);
      if (number_of_algebraic_constraints > kmcdSurf.nSpec) {
        number_of_algebraic_constraints = kmcdSurf.nSpec;

        printf("------------------------------------------------------------------------- \n");
        printf("-----------------------------------WARNING------------------------------- \n");
        printf(" Number of algebraic constraints is bigger than number of surface species \n");
        printf(" Setting number of algebraic constrains equal to number of surface species %d \n", kmcdSurf.nSpec);
        printf("------------------------------------------------------------------------- \n");
        printf("------------------------------------------------------------------------- \n");

      }

      cstr.number_of_algebraic_constraints = number_of_algebraic_constraints;

      cstr.Yi = real_type_1d_view("Mass fraction at inlet", kmcd.nSpec);
      printf("Reactor residence time [s] %e\n", state_host_inlet(0,0)*cstr.Vol/cstr.mdotIn);
      cstr.poisoning_species_idx=poisoning_species_idx;

      // work batch = 1
      Kokkos::parallel_for(
        Kokkos::RangePolicy<TChem::exec_space>(0, nBatch),
        KOKKOS_LAMBDA(const ordinal_type& i) {
          //mass fraction
          for (ordinal_type k = 0; k < kmcd.nSpec; k++) {
            cstr.Yi(k) = state_inlet(0,k+3);
          }
      });

      real_type_2d_view EnthalpyMass("EnthalpyMass", 1, kmcd.nSpec);
      real_type_1d_view EnthalpyMixMass("EnthalpyMass Mixture", 1);

      const auto exec_space_instance = TChem::exec_space();
      TChem::EnthalpyMass::runDeviceBatch(exec_space_instance,
                                            team_size,
                                            vector_size,
                                            1,
                                            state_inlet,
                                            EnthalpyMass,
                                            EnthalpyMixMass,
                                            kmcd);
      cstr.EnthalpyIn = EnthalpyMixMass(0);
    }


    using problem_type =
    TChem::Impl::TransientContStirredTankReactor_Problem<real_type,device_type>;

    const ordinal_type n_equations = problem_type::getNumberOfTimeODEs(kmcd, kmcdSurf, cstr);
    real_type_2d_view rhs("RHS", nBatch, n_equations);

    /// team policy
    policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());
    const ordinal_type level = 1;

    if (team_size > 0 && vector_size > 0) {
      policy = policy_type(exec_space_instance, nBatch, team_size, vector_size);
    }
    const ordinal_type per_team_extent = TChem::
    IsothermalTransientContStirredTankReactorRHS::getWorkSpaceSize(kmcd, kmcdSurf);

    const ordinal_type per_team_scratch =
      TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
    policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

    TChem::
    IsothermalTransientContStirredTankReactorRHS::runDeviceBatch( policy, state,
       site_fraction, rhs,  kmcds, kmcdSurfs, cstr);

    // save all samples for rhs i
    auto rhs_of_ivar = Kokkos::subview(rhs, Kokkos::ALL(), ivar);
    auto rhs_of_ivar_host = Kokkos::create_mirror_view(rhs_of_ivar);
    Kokkos::deep_copy(rhs_of_ivar_host, rhs_of_ivar);
    TChem::Test::writeReactionRates(outputFile, nBatch, rhs_of_ivar_host);

  }
  Kokkos::finalize();

  #else
   printf("This example requires Yaml ...\n" );
  #endif

  return 0;

}
