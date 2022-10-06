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
#ifndef __TCHEM_TEST_SURFACE_FORD_MOTZWISE_HPP__
#define __TCHEM_TEST_SURFACE_FORD_MOTZWISE_HPP__

// Units test for FORD type reactions and motz-wise correlation

TEST(SurfaceFordMotzWise, single)
{
  std::string exec="../../example/TChem_NetProductionRateSurfacePerMass.x";
  std::string prefixPath="../../example/data/surface-ford-motz-wise/";


  std::string invoke=(exec +" "+
           "--chemfile="+ prefixPath +"chem.yaml "+
           "--inputfile=" + prefixPath +"inputGas.dat "+
           "--inputfileSurf=" + prefixPath +"inputSurfGas.dat "+
           "--outputfile=omega.dat "+
           "--outputfileGasSurf=omegaGasSurf.dat "+
           "--outputfileSurf=omegaSurf.dat "+
           "--use-yaml=true");

  const auto invoke_c_str = invoke.c_str();
  printf("testing : %s\n", invoke_c_str);
  std::system(invoke_c_str);

  /// compare with ref
  EXPECT_TRUE(TChem::Test::compareFilesValues("omega.dat",
          "reference/omega.dat"));
  EXPECT_TRUE(TChem::Test::compareFilesValues("omegaGasSurf.dat",
					"reference/omegaGasSurf.dat"));
  EXPECT_TRUE(TChem::Test::compareFilesValues("omegaSurf.dat",
          "reference/omegaSurf.dat"));

  /// check echo file
  EXPECT_TRUE(TChem::Test::compareFiles("kmod.echo","reference/kmod.echo"));
  EXPECT_TRUE(TChem::Test::compareFiles("kmodSurf.echo","reference/kmodSurf.echo"));


}

#endif
