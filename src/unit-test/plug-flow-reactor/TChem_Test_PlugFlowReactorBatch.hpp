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
#ifndef __TCHEM_TEST_PlUGFLOWREACTORBATCH_HPP__
#define __TCHEM_TEST_PlUGFLOWREACTORBATCH_HPP__


TEST(PlugFlowReactorBatch, single)
{
  std::string exec="../../example/TChem_PlugFlowReactor.x";
  std::string prefixPath="../../example/data/plug-flow-reactor/CH4-PTnogas/";
  std::string max_z_iterations="4";
  std::string use_prefixPath="false";
  std::string inputsPath = "../../example/runs/PlugFlowReactor/CH4-PTnogas_SA/inputs/";

  std::string chemFile(prefixPath + "chem.inp");
  std::string thermFile(prefixPath + "therm.dat");
  std::string chemSurfFile  = prefixPath + "chemSurf.inp";
  std::string thermSurfFile = prefixPath + "thermSurf.dat";
  std::string inputFile     = inputsPath + "sample.dat";
  std::string inputFileSurf = inputsPath + "inputSurf.dat";
  std::string inputFilevelocity = inputsPath + "inputVelocity.dat";
  std::string outputFile("PFRSolution8Samples.dat");

  std::string transient_initial_condition="true";
  std::string initial_condition="false";

  std::string atol_newton="1e-14";
  std::string rtol_newton="1e-8";
  std::string tol_z="1e-8";

  std::string invoke=(exec +" "+
           "--inputs-path="+ prefixPath +" "+
           "--chemfile=" + chemFile +" "+
 		       "--thermfile=" + thermFile +" "+
           "--surf-chemfile=" + chemSurfFile +" "+
           "--surf-thermfile=" + thermSurfFile +" "+
           "--samplefile=" + inputFile +" "+
           "--surf-inputfile=" + inputFileSurf +" "+
           "--velocity-inputfile=" + inputFilevelocity +" "+
           "--max-z-iterations=" + max_z_iterations +" "+
           "--outputfile=" + outputFile +" "+
           "--transient-initial-condition=" + transient_initial_condition +" "+
           "--initial-condition="+ initial_condition +" "+
           "--atol-newton=" + atol_newton +" "+
           "--rtol-newton=" + rtol_newton +" "+
           "--tol-z=" +tol_z);

  const auto invoke_c_str = invoke.c_str();
  printf("testing : %s\n", invoke_c_str);
  std::system(invoke_c_str);

  /// compare with ref
  EXPECT_TRUE(TChem::Test::compareFilesValues(outputFile,
					"reference/"+outputFile));

}

#endif
