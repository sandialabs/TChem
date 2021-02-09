/* =====================================================================================
TChem version 2.1.0
Copyright (2020) NTESS
https://github.com/sandialabs/TChem

Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
certain rights in this software.

This file is part of TChem. TChem is open-source software: you can redistribute it
and/or modify it under the terms of BSD 2-Clause License
(https://opensource.org/licenses/BSD-2-Clause). A copy of the license is also
provided under the main directory

Questions? Contact Cosmin Safta at <csafta@sandia.gov>, or
           Kyungjoo Kim at <kyukim@sandia.gov>, or
           Oscar Diaz-Ibarra at <odiazib@sandia.gov>

Sandia National Laboratories, Livermore, CA, USA
===================================================================================== */


#ifndef __TCHEM_TEST_IGNITIONZEROD_HPP__
#define __TCHEM_TEST_IGNITIONZEROD_HPP__


TEST(IgnitionZeroD, single)
{
  std::string exec="../../example/TChem_IgnitionZeroDSA.x";
  std::string prefixPath="../../example/data/ignition-zero-d/gri3.0/";
  std::string max_time_iterations="4";
  std::string save="1";

  std::string invoke=(exec +" "+
           "--inputsPath="+ prefixPath +" "+
           "--max-time-iterations=" + max_time_iterations +" "+
           "--output_frequency=" + save);

  const auto invoke_c_str = invoke.c_str();
  printf("testing : %s\n", invoke_c_str);
  std::system(invoke_c_str);

  /// compare with ref
  EXPECT_TRUE(TChem::Test::compareFilesValues("IgnSolution.dat",
					"reference/IgnSolution.dat"));

  /// check echo file
  EXPECT_TRUE(TChem::Test::compareFiles("kmod.echo","reference/kmod.echo"));
  EXPECT_TRUE(TChem::Test::compareFiles("kmod.list","reference/kmod.list"));

}

#endif
