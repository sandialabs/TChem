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
#ifndef __TCHEM_TEST_IGNITIONZEROD_HPP__
#define __TCHEM_TEST_IGNITIONZEROD_HPP__


TEST(IgnitionZeroD, single)
{
  std::string exec="../../example/TChem_IgnitionZeroD.x";
  std::string prefixPath="../../example/data/ignition-zero-d/gri3.0/";
  std::string max_time_iterations="4";
  std::string atol_newton="1e-18";
  std::string rtol_newton="1e-8";
  std::string tol_time="1e-8";
  std::string dtmin="1e-20";
  std::string dtmax="1e-3";

  std::string invoke=(exec +" "+
           "--inputs-path="+ prefixPath +" "+
           "--max-time-iterations=" + max_time_iterations +" "+
           "--use-prefix-path=true " +
           "--atol_newton=" + atol_newton +" "+
           "--rtol_newton=" + rtol_newton +" "+
           "--tol_time=" +tol_time +" "+
           "--dtmin=" + dtmin + " "+
           "--dtmax=" + dtmax);

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

TEST(SourceAndJacobians, single)
{
  std::string exec="../../example/TChem_IgnitionZeroD_Jacobians.x";
  std::string prefixPath="../../example/data/ignition-zero-d/gri3.0/";

  std::string chemfile(prefixPath + "chem.inp");
  std::string thermfile(prefixPath + "therm.dat");
  std::string inputfile(prefixPath + "inputGas.dat");
  std::string outputfile("ign_zero_d.dat");

  std::string invoke = (exec +
                        " --batchsize=1"+
                        " --outputfile="+outputfile +
                        " --verbose=true" +
                        " --use-sample-format=false"+
                        " --output-file-times=_wall_times.dat" +
                        " --chemfile="  + chemfile +
                        " --thermfile=" + thermfile +
                        " --inputfile=" + inputfile);

  const auto invoke_c_str = invoke.c_str();
  printf("testing : %s\n", invoke_c_str);
  std::system(invoke_c_str);

  // /// compare with ref
  EXPECT_TRUE(TChem::Test::compareFilesValues("source_term_ign_zero_d.dat",
					"reference/source_term_ign_zero_d.dat"));
  //
  EXPECT_TRUE(TChem::Test::compareFilesValues("analytic_Jacobian_ign_zero_d.dat",
					"reference/analytic_Jacobian_ign_zero_d.dat"));
  //
  EXPECT_TRUE(TChem::Test::compareFilesValues("numerical_Jacobian_ign_zero_d.dat",
					"reference/numerical_Jacobian_ign_zero_d.dat"));
  //
  EXPECT_TRUE(TChem::Test::compareFilesValues("sacado_Jacobian_ign_zero_d.dat",
					"reference/sacado_Jacobian_ign_zero_d.dat"));
  //
  EXPECT_TRUE(TChem::Test::compareFilesValues("numerical_Jacobian_fwd_ign_zero_d.dat",
					"reference/numerical_Jacobian_fwd_ign_zero_d.dat"));

}

#endif
