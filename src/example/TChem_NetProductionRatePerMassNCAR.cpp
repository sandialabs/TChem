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
#include "TChem_Impl_KForward.hpp"
#include "TChem_CommandLineParser.hpp"
#include "TChem.hpp"
#include "TChem_Impl_ReactionRatesAerosol.hpp"
#include "TChem_AtmosphericChemistry.hpp"

using ordinal_type = TChem::ordinal_type;
using real_type = TChem::real_type;
using real_type_1d_view = TChem::real_type_1d_view;
using real_type_2d_view = TChem::real_type_2d_view;

int
main(int argc, char* argv[])
{
  #if defined(TCHEM_ENABLE_TPL_YAML_CPP)
  /// default inputs
  std::string prefixPath("");
  std::string chemFile(prefixPath + "chem.yaml");
  std::string inputFile(prefixPath + "input.dat");
  std::string outputFile(prefixPath + "omega.dat");
  std::string thermFile(prefixPath + "therm.dat");
  int nBatch(1);
  bool verbose(true);
  bool useYaml(true);
  bool use_sample_format(false);
  std::string test("arrhenius");

  /// parse command line arguments
  TChem::CommandLineParser opts(
    "This example computes reaction rates with a given state vector");
  opts.set_option<std::string>(
    "chemfile", "Chem file name e.g., chem.yaml", &chemFile);
  opts.set_option<std::string>(
    "inputfile", "Input state file name e.g., input.dat", &inputFile);
  opts.set_option<std::string>(
    "outputfile", "Output omega file name e.g., omega.dat", &outputFile);
  opts.set_option<std::string>(
    "thermfile", "Therm file name e.g., therm.dat", &thermFile);
  //
  opts.set_option<std::string>(
    "unit-test", "unit test name e.g., arrhenius", &test);
  opts.set_option<int>(
    "batchsize",
    "Batchsize the same state vector described in statefile is cloned",
    &nBatch);
  opts.set_option<bool>(
    "verbose", "If true, printout the first omega values", &verbose);
  opts.set_option<bool>(
    "useYaml", "If true, use yaml to parse input file", &useYaml);
  opts.set_option<bool>(
    "use_sample_format", "If true, input file does not header or format", &use_sample_format);

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse)
    return 0; // print help return

  Kokkos::initialize(argc, argv);
  {
    const bool detail = false;

    TChem::exec_space().print_configuration(std::cout, detail);
    TChem::host_exec_space().print_configuration(std::cout, detail);

    using host_device_type      = typename Tines::UseThisDevice<host_exec_space>::type;

    /// construct kmd and use the view for testing
    TChem::KineticModelData kmd = TChem::KineticModelData(chemFile);
    const auto kmcd = TChem::createNCAR_KineticModelConstData<host_device_type>(kmd);
    const auto member = Tines::HostSerialTeamMember();

    const auto speciesNamesHost = Kokkos::create_mirror_view(kmcd.speciesNames);
    Kokkos::deep_copy(speciesNamesHost, kmcd.speciesNames);


    // forward constant
    using kForward_type = TChem::Impl::KForward<real_type, host_device_type >;
    using value_type_1d_view_type = typename kForward_type::value_type_1d_view_type;

    value_type_1d_view_type kfor("forward reate constant", kmcd.nReac);

    using reaction_rates_type = TChem::Impl::ReactionRatesAerosol<real_type, host_device_type >;

    value_type_1d_view_type omega("omega", kmcd.nSpec);

    value_type_1d_view_type work("work", 3*kmcd.nReac);

    // read scenario condition from yaml file
    real_type_2d_view_host state_host;
    TChem::AtmChemistry
         ::setScenarioConditions(chemFile, speciesNamesHost,
                                 kmcd.nSpec, state_host, nBatch );

    const auto t = state_host(0,2);
    const auto p = state_host(0,1);

    kForward_type::team_invoke(member, t, p, kfor, kmcd);

    TChem::Test::writeReactionRates(outputFile, kmcd.nReac, kfor);



    // // arrhenius test
    // if(test=="arrhenius"){
    //
    //   printf("Testing arrhenius type\n");
    //   const real_type t(272.5);
    //   const real_type p(101253.3);
    //
    //   kForward_type::team_invoke( member, t, p, kfor,  kmcd);
    //
    //   for (size_t i = 0; i < kmcd.nReac; i++) {
    //     printf(" kfor %e\n",kfor(i) );
    //   }
    //   value_type_1d_view_type state("state", kmcd.nSpec-kmcd.nConstSpec);
    //   state(0)= real_type(1.0); // A
    //   state(1)= real_type(1e-60); // B
    //   state(2)= real_type(1e-60); // C
    //
    //
    //   value_type_1d_view_type const_state("const state", kmcd.nConstSpec);
    //
    //   const real_type conv = CONV_PPM * p /t ; //D
    //   state(0)= real_type(1.2)/conv; // D
    //
    //   reaction_rates_type::team_invoke(member, t,  p, state, const_state, omega, work,  kmcd);
    //
    //   for (size_t i = 0; i < kmcd.nSpec; i++) {
    //     printf(" omega %e\n",omega(i) );
    //   }
    // } else if(test=="troe")
    // //troe test
    // {
    //   printf("Testing troe type\n");
    //
    //   const real_type t(272.5);
    //   const real_type p(101253.3);
    //
    //   kForward_type::team_invoke( member, t, p, kfor, kmcd);
    //
    //   for (size_t i = 0; i < kmcd.nReac; i++) {
    //     printf(" kfor %e\n",kfor(i) );
    //   }
    //   value_type_1d_view_type state("state", kmcd.nSpec-kmcd.nConstSpec);
    //   state(0)= real_type(0.5); // A
    //   state(1)= real_type(0.6); // B
    //   state(2)= real_type(0.3); // C
    //
    //   value_type_1d_view_type const_state("const state", kmcd.nConstSpec);
    //
    //   reaction_rates_type::team_invoke(member, t,  p, state, const_state, omega, work,  kmcd);
    //
    //   for (size_t i = 0; i < kmcd.nSpec; i++) {
    //     printf(" omega %e\n",omega(i) );
    //   }
    //
    // }













  }
  Kokkos::finalize();

  #else
   printf("This example requires Yaml ...\n" );
  #endif

  return 0;
}
