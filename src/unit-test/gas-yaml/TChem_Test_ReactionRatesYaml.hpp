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
#ifndef __TCHEM_TEST_REACTIONRATESYAML_HPP__
#define __TCHEM_TEST_REACTIONRATESYAML_HPP__

// #include "TChem_KineticModelData.hpp"
// #include "TChem_NetProductionRatePerMass.hpp"

TEST(NetProductionRatePerMassYaml, single) {
#if defined(TCHEM_ENABLE_TPL_YAML_CPP)
  std::string exec = "../../example/TChem_NetProductionRatePerMass_Yaml.x";

  std::string prefixPath = "../../example/data/reaction-rates/";
  std::string chemFile(prefixPath + "chem.yaml");
  // std::string thermFile(prefixPath + "therm.dat");
  std::string inputFile(prefixPath + "input.dat");
  std::string outputFile("net-production-rate-per-mass.dat");
  std::string invoke =
      (exec + " " + "--chemfile=" + chemFile + " " +
       // "--thermfile="+thermFile+" "+
       "--inputfile=" + inputFile + " " + "--outputfile=" + outputFile + " " + "--batchsize=1 verbose=true");
  const auto invoke_c_str = invoke.c_str();
  printf("testing : %s\n", invoke_c_str);
  std::system(invoke_c_str);

  /// compare with ref
  EXPECT_TRUE(TChem::Test::compareFilesValues("net-production-rate-per-mass.dat",
                                              "../reference/net-production-rate-per-mass.dat"));
//
#else
  printf("This test requires Yaml ...\n");
#endif
}

TEST(ChebyshevReactionType, single) {
#if defined(TCHEM_ENABLE_TPL_YAML_CPP)
  std::string exec = "../../example/TChem_NetProductionRatePerMass_Yaml.x";

  std::string prefixPath = "inputs_chebyshev/";
  std::string chemFile(prefixPath + "chem.yaml");
  std::string inputFile(prefixPath + "inputGas.dat");
  std::string outputFile("net-production-rate-per-mass-chebyshev.dat");
  std::string invoke = (exec + " " + "--chemfile=" + chemFile + " " + "--inputfile=" + inputFile + " " +
                        "--outputfile=" + outputFile + " " + "--batchsize=1 verbose=true");
  const auto invoke_c_str = invoke.c_str();
  printf("testing : %s\n", invoke_c_str);
  std::system(invoke_c_str);

  EXPECT_TRUE(TChem::Test::compareFilesValues("net-production-rate-per-mass-chebyshev.dat",
                                              prefixPath + "net-production-rate-per-mass-chebyshev.dat"));

  EXPECT_TRUE(TChem::Test::compareFilesValues("ropfwd_net-production-rate-per-mass-chebyshev.dat",
                                              prefixPath + "ropfwd_net-production-rate-per-mass-chebyshev.dat"));

  EXPECT_TRUE(TChem::Test::compareFilesValues("roprev_net-production-rate-per-mass-chebyshev.dat",
                                              prefixPath + "roprev_net-production-rate-per-mass-chebyshev.dat"));

  EXPECT_TRUE(TChem::Test::compareFilesValues("kfwd_net-production-rate-per-mass-chebyshev.dat",
                                              prefixPath + "kfwd_net-production-rate-per-mass-chebyshev.dat"));

  EXPECT_TRUE(TChem::Test::compareFilesValues("krev_net-production-rate-per-mass-chebyshev.dat",
                                              prefixPath + "krev_net-production-rate-per-mass-chebyshev.dat"));

#else
  printf("This test requires Yaml ...\n");
#endif
}

#endif
