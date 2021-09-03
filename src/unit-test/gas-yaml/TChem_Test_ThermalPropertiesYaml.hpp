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
#ifndef __TCHEM_TEST_THERMALPROPERTIESYAML_HPP__
#define __TCHEM_TEST_THERMALPROPERTIESYAML_HPP__

TEST(ThermalPropertiesYaml, single)
{
  #if defined(TCHEM_ENABLE_TPL_YAML_CPP)
  std::string exec="../../example/TChem_ThermalProperties_Yaml.x";

  std::string prefixPath="../../example/data/reaction-rates/";
  std::string chemFile(prefixPath + "chem.yaml");
  std::string inputFile("../../example/runs/ThermalProperties/sample.dat");
  std::string invoke=(exec+" "+
		      "--chemfile="+chemFile+" "+
		      "--inputfile="+inputFile+" "+
		      "--batchsize=1 verbose=true");
  const auto invoke_c_str = invoke.c_str();
  printf("testing : %s\n", invoke_c_str);
  std::system(invoke_c_str);

  /// compare with ref
  EXPECT_TRUE(TChem::Test::compareFiles("CpMixMass.dat",
					"../reference/CpMixMass.dat"));
  /// compare with ref
  EXPECT_TRUE(TChem::Test::compareFiles("CvMixMass.dat",
					"../reference/CvMixMass.dat"));
  /// compare with ref
  EXPECT_TRUE(TChem::Test::compareFiles("EnthalpyMixMass.dat",
          "../reference/EnthalpyMixMass.dat"));
  /// compare with ref
  EXPECT_TRUE(TChem::Test::compareFiles("EntropyMixMass.dat",
          "../reference/EntropyMixMass.dat"));
  /// compare with ref
  EXPECT_TRUE(TChem::Test::compareFiles("InternalEnergyMixMass.dat",
          "../reference/InternalEnergyMixMass.dat"));

  //
  #else
   printf("This test requires Yaml ...\n" );
  #endif
}

#endif
