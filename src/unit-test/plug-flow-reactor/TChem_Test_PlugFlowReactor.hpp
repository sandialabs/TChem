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
#ifndef __TCHEM_TEST_PlUGFLOWREACTOR_HPP__
#define __TCHEM_TEST_PlUGFLOWREACTOR_HPP__


TEST(PlugFlowReactor, single)
{
  std::string exec="../../example/TChem_PlugFlowReactor.x";
  std::string prefixPath="../../example/data/plug-flow-reactor/CH4-PTnogas/";
  std::string max_z_iterations="4";
  std::string save="1";
  std::string transient_initial_condition="true";
  std::string initial_condition="false";

  std::string invoke=(exec +" "+
           "--prefixPath="+ prefixPath +" "+
           "--max-z-iterations=" + max_z_iterations +" "+
           "--transient_initial_condition=" + transient_initial_condition +" "+
           "--initial_condition="+ initial_condition +" "
           "--output_frequency=" + save);

  const auto invoke_c_str = invoke.c_str();
  printf("testing : %s\n", invoke_c_str);
  std::system(invoke_c_str);

  /// compare with ref
  EXPECT_TRUE(TChem::Test::compareFilesValues("PFRSolution.dat",
					"reference/PFRSolution.dat"));

  /// check echo file
  EXPECT_TRUE(TChem::Test::compareFiles("kmod.echo","reference/kmod.echo"));
  EXPECT_TRUE(TChem::Test::compareFiles("kmod.list","reference/kmod.list"));

  EXPECT_TRUE(TChem::Test::compareFiles("kmodSurf.echo","reference/kmodSurf.echo"));
  EXPECT_TRUE(TChem::Test::compareFiles("kmodSurf.list","reference/kmodSurf.list"));

}

#endif
